"""Native Qwen forwards for one-matrix editing; the notebook defines the mathematics."""

import hashlib
import json
import os
import re
import unicodedata
from importlib.metadata import version
from functools import lru_cache
from pathlib import Path
from time import perf_counter

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("EASYDEL_AUTO", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
from easydel.layers.rotary_embedding import compute_basic_frequencies
from eformer import common_types
from ejkernel.types import MaskInfo
from flax import nnx
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from jax.flatten_util import ravel_pytree
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SYSTEM = "You are a helpful assistant."
STOP_IDS = (151645, 151643)


def load_model():
    """Load the pinned official checkpoint with native, differentiable attention."""
    jax.config.update("jax_default_matmul_precision", "highest")
    files = ["config.json", "generation_config.json", "model.safetensors",
             "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    try:
        snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        snapshot = None
    if snapshot is None or not all((Path(snapshot) / name).is_file() for name in files):
        snapshot = snapshot_download(MODEL, revision=REVISION, allow_patterns=files,
                                     etag_timeout=15)
    # Local paths also avoid unrelated remote metadata calls in Transformers 4.57.
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, from_torch=True,
        dtype=jnp.float32, param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST, platform="jax",
        auto_shard_model=False, sharding_axis_dims=(1, 1, 1, 1, 1),
        quantization_config=None, quantize_tensors=False, torch_load_mode="streaming",
        config_kwargs=dict(attn_mechanism="vanilla", attn_dtype=jnp.float32,
                           attn_softmax_dtype=jnp.float32, kvdtype=jnp.float32,
                           gradient_checkpointing="", mask_max_position_embeddings=256,
                           freq_max_position_embeddings=256),
    )
    # EasyDeL 0.2.0.2 rounds this nonparameter cache to bfloat16 by default.
    # Recompute with its native helper; upcasting rounded values would not fix parity.
    assert model.config.rope_scaling is None
    model.model.frequencies.value = compute_basic_frequencies(
        base=model.config.rope_theta, rotary_dim=model.config.head_dim,
        max_position_embeddings=256,
    )
    assert model.config.tie_word_embeddings
    assert model.config.attention_dropout == model.config.resid_pdrop == 0
    model.eval()
    return model, tokenizer


@lru_cache(maxsize=1)
def load_editor(layer=20):
    """Load once per kernel; keep native-loader logs accessible on failure."""
    from IPython.utils.capture import capture_output

    print(f"Loading {MODEL}…")
    try:
        with capture_output() as loading:
            model, tokenizer = load_model()
            editor = select_matrix(model, layer=layer)
    except Exception:
        loading.show()
        raise
    print(f"Device: {jax.devices()[0]} · editable matrix: {editor['shape']} · float32")
    return editor, tokenizer


def select_matrix(model, layer=20):
    """Expose only down_proj.kernel; its stored shape is (input, output)."""
    if not 0 <= layer < model.config.num_hidden_layers:
        raise ValueError("layer must be a zero-based transformer-block index")
    path = ("model", "layers", layer, "mlp", "down_proj", "kernel")
    shape = tuple(model.model.layers[layer].mlp.down_proj.kernel.value.shape)
    assert shape == (model.config.intermediate_size, model.config.hidden_size), (
        f"Expected down_proj stored INPUT x OUTPUT, obtained {shape}")
    graph, parameters, fixed = nnx.split(
        model, lambda p, value: p == path and isinstance(value, nnx.Param), ...
    )
    theta0, unravel = ravel_pytree(parameters)
    assert len(jax.tree_util.tree_leaves(parameters)) == 1
    assert theta0.size == np.prod(shape)
    mesh = model.mesh

    @jax.jit
    def full(theta, fixed, ids, mask, positions, query):
        with mesh:
            m = nnx.merge(graph, unravel(theta), fixed)
            hidden = m.model(ids, attention_mask=mask, position_ids=positions).last_hidden_state
            return m.apply_lm_head(hidden[jnp.arange(len(ids))[:, None], query])

    @jax.jit
    def prefix(theta, fixed, ids, mask, positions):
        with mesh:
            m = nnx.merge(graph, unravel(theta), fixed)
            output = m.model(ids, attention_mask=mask, position_ids=positions,
                             output_hidden_states=True)
            return output.hidden_states[layer]  # Input to the selected block; wholly frozen.

    @jax.jit
    def suffix(theta, fixed, hidden, mask, positions, query):
        with mesh:
            m = nnx.merge(graph, unravel(theta), fixed)
            info = MaskInfo.dynamic_init(inputs_embeds=hidden, attention_mask=mask)
            # This is the actual nonlinear model, including all downstream blocks.
            for block in m.model.layers[layer:]:
                hidden = block(hidden, info, positions, mode=common_types.MODE_TRAIN,
                               frequencies=m.model.frequencies).hidden_states
            hidden = m.model.norm(hidden)
            return m.apply_lm_head(hidden[jnp.arange(len(hidden))[:, None], query])

    def projection_input(m, hidden, mask, positions):
        # Invoke native components to observe the input of this exact projection.
        # No attention, normalization, activation, or rotary formula is reimplemented.
        block = m.model.layers[layer]
        info = MaskInfo.dynamic_init(inputs_embeds=hidden, attention_mask=mask)
        attention = block.self_attn(block.input_layernorm(hidden), info, positions,
                                    mode=common_types.MODE_TRAIN, frequencies=m.model.frequencies)
        mlp_input = block.post_attention_layernorm(hidden + attention.attention_output)
        keys = block.mlp.act_fn(block.mlp.gate_proj(mlp_input)) * block.mlp.up_proj(mlp_input)
        return keys, mlp_input

    @jax.jit
    def activations(theta, fixed, ids, mask, positions):
        with mesh:
            m = nnx.merge(graph, unravel(theta), fixed)
            hidden = prefix(theta, fixed, ids, mask, positions)
            return projection_input(m, hidden, mask, positions)[0]

    @jax.jit
    def activation_error(theta, fixed, ids, mask, positions):
        """Check that observed keys reconstruct the native MLP output exactly."""
        with mesh:
            m = nnx.merge(graph, unravel(theta), fixed)
            hidden = prefix(theta, fixed, ids, mask, positions)
            keys, mlp_input = projection_input(m, hidden, mask, positions)
            mlp = m.model.layers[layer].mlp
            return jnp.max(jnp.abs(mlp(mlp_input) - mlp.down_proj(keys)))

    return dict(theta0=theta0, shape=shape, fixed=fixed, layer=layer, revision=REVISION,
                full=full, prefix=prefix, suffix=suffix, activations=activations,
                activation_error=activation_error)


def chat_ids(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True,
    )


def subject_activations(editor, tokenizer, contexts, subject, length=128):
    """Original down_proj inputs at the last subject token, using text offsets.

    Contexts are strings or records with ``text`` and optional ``subject_span``
    (character start/end). Ambiguous repeated subjects require an explicit span.
    The real chat template is used, but its assistant prefix is never the key.
    """
    if not contexts or not 1 <= length <= 256:
        raise ValueError("Supply nonempty key contexts and length in [1,256]")
    ids = np.full((len(contexts), length), tokenizer.pad_token_id, np.int32)
    mask = np.zeros_like(ids)
    records = []
    for row, context in enumerate(contexts):
        record = {"text": context} if isinstance(context, str) else context
        text = record.get("text", record.get("context", record.get("question")))
        if not isinstance(text, str):
            raise ValueError("Each key context needs text")
        name = record.get("subject", subject)
        matches = list(re.finditer(re.escape(name), text, flags=re.IGNORECASE))
        if "subject_span" in record:
            start, end = record["subject_span"]
            if text[start:end].casefold() != name.casefold():
                raise ValueError("subject_span does not identify the supplied subject")
        elif len(matches) == 1:
            start, end = matches[0].span()
        else:
            raise ValueError("Key context must contain one subject or an explicit subject_span")
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
        if rendered.count(text) != 1:
            raise ValueError("Cannot locate the unique user text in the chat template")
        offset = rendered.index(text)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        tokens, spans = encoded["input_ids"], encoded["offset_mapping"]
        assert tokens == chat_ids(tokenizer, text)
        indices = [i for i, (left, right) in enumerate(spans)
                   if left < offset + end and right > offset + start]
        if not indices or len(tokens) > length:
            raise ValueError("Subject is unaligned or key context exceeds the fixed length")
        index = indices[-1]
        assert spans[index][0] < offset + end <= spans[index][1]
        ids[row, :len(tokens)], mask[row, :len(tokens)] = tokens, 1
        records.append(dict(id=record.get("id", str(row)), text=text, subject=name,
                            subject_span=[start, end], token_index=index,
                            token_id=tokens[index], token_piece=tokenizer.decode([tokens[index]])))
    positions = np.maximum(mask.cumsum(-1, dtype=np.int32) - 1, 0)
    keys = editor["activations"](editor["theta0"], editor["fixed"], ids, mask, positions)
    selected = keys[jnp.arange(len(records)), jnp.array([r["token_index"] for r in records])]
    average = jnp.mean(selected, axis=0)
    if not bool(jnp.all(jnp.isfinite(average))) or float(jnp.linalg.norm(average)) == 0:
        raise FloatingPointError("The original subject activation must be finite and nonzero")
    return average, records


def edit_direction(editor, tokenizer, subject, geometry="euclidean"):
    """Unit subject direction; optional activation geometry uses original reference data.

    The five neutral contexts contain the subject, never the edited answer.
    Euclidean mode needs no reference corpus or covariance calculation.
    """
    if not subject.strip() or geometry not in ("euclidean", "activation"):
        raise ValueError("Supply a subject and geometry='euclidean' or 'activation'")
    contexts = [f"The {subject}", f"A visitor asked about the {subject}",
                f"This article describes the {subject}", f"Many travelers recognize the {subject}",
                f"Let us discuss the {subject}"]
    a, _ = subject_activations(editor, tokenizer, contexts, subject, length=128)
    if geometry == "euclidean":
        z = a
    else:
        moment, _ = covariance(editor, tokenizer, exclude_subjects=[subject])
        metric = moment.astype(np.float64)
        metric[np.diag_indices_from(metric)] += 1e-3
        key = np.asarray(a, dtype=np.float64)
        solved = np.linalg.solve(metric, key)
        error = np.linalg.norm(metric @ solved - key) / np.linalg.norm(key)
        if not np.isfinite(error) or error > 1e-3:
            raise FloatingPointError(f"Activation-metric solve failed: relative error {error}")
        z = jnp.asarray(solved, dtype=jnp.float32)
    b = z / (a @ z)
    return b / jnp.linalg.norm(b)


def answer_logits(editor, theta, fixed, batch):
    """Real Qwen logits only at supplied-answer positions, including chat EOS."""
    return editor["suffix"](theta, fixed, batch["hidden"], batch["mask"],
                             batch["positions"], batch["query"])


def prepare_edit(editor, tokenizer, questions, answers, length=None, answer_length=None):
    """Score only supplied answer tokens and chat EOS, with automatic padded shapes.

    Multi-token answers use the official completed chat template. No generation
    supplies targets, and the frozen prefix is rebuilt for every preparation.
    """
    questions = [questions] if isinstance(questions, str) else list(questions)
    answers = [answers] if isinstance(answers, str) else list(answers)
    if not questions or len(questions) != len(answers):
        raise ValueError("Supply the same nonzero number of questions and answers")
    sequences = []
    for question, answer in zip(questions, answers, strict=True):
        if not question.strip() or not answer.strip() or any(
                token in answer for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
            raise ValueError("Supply nonempty question/answer text without answer special tokens")
        prompt = chat_ids(tokenizer, question)
        complete = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question},
             {"role": "assistant", "content": answer}],
            tokenize=True, add_generation_prompt=False,
        )
        if complete[:len(prompt)] != prompt:
            raise ValueError("Completed chat does not share the official generation prefix")
        end = complete.index(STOP_IDS[0], len(prompt)) + 1
        answer_ids = complete[len(prompt):end]  # Include EOS, exclude template newline.
        assert len(answer_ids) >= 2 and answer_ids[-1] == STOP_IDS[0]
        sequences.append((prompt, answer_ids))
    needed = max(len(prompt) + len(answer) - 1 for prompt, answer in sequences)
    if needed + 1 > 256:
        raise ValueError("Prompt plus answer including EOS must fit this tutorial's 256-token limit")
    length = max(64, 32 * ((needed + 31) // 32)) if length is None else length
    answer_length = max(len(answer) for _, answer in sequences) if answer_length is None else answer_length
    if not 1 <= length <= 256 or answer_length < 1:
        raise ValueError("This small tutorial supports at most 256 tokens per training sequence")
    if needed > length or any(len(answer) > answer_length for _, answer in sequences):
        raise ValueError("Edit exceeds explicit length/answer_length; omit them for automatic sizing")
    ids = np.full((len(questions), length), tokenizer.pad_token_id, np.int32)
    mask = np.zeros_like(ids)
    query = np.zeros((len(questions), answer_length), np.int32)
    targets = np.full(query.shape, STOP_IDS[0], np.int32)
    valid = np.zeros(query.shape, bool)
    for i, (prompt, answer_ids) in enumerate(sequences):
        sequence = prompt + answer_ids[:-1]
        ids[i, :len(sequence)], mask[i, :len(sequence)] = sequence, 1
        query[i] = len(prompt) - 1 + np.minimum(np.arange(answer_length), len(answer_ids) - 1)
        targets[i, :len(answer_ids)], valid[i, :len(answer_ids)] = answer_ids, True
    positions = np.maximum(mask.cumsum(-1, dtype=np.int32) - 1, 0)
    hidden = editor["prefix"](editor["theta0"], editor["fixed"], ids, mask, positions)
    return dict(hidden=hidden, input_ids=jnp.asarray(ids), mask=jnp.asarray(mask),
                positions=jnp.asarray(positions), query=jnp.asarray(query),
                answer_ids=jnp.asarray(targets), valid=jnp.asarray(valid),
                valid_indices=jnp.asarray(np.flatnonzero(valid.reshape(-1)), dtype=jnp.int32))


@jax.jit
def _sample_tokens(logits, keys, temperature, top_p):
    """Full-vocabulary nucleus sampling, retaining the threshold-crossing token."""
    order = jnp.argsort(logits, axis=-1, descending=True)
    ordered = jnp.take_along_axis(logits / temperature, order, axis=-1)
    probabilities = jax.nn.softmax(ordered, axis=-1)
    keep = jnp.cumsum(probabilities, axis=-1) - probabilities < top_p
    truncated = jnp.where(keep, ordered, -jnp.inf)
    indices = jax.vmap(jax.random.categorical)(keys, truncated)
    return order[jnp.arange(len(order)), indices]


def generate(editor, tokenizer, theta, questions, max_new_tokens=24, batch_size=4,
             length=128, *, seeds=None, temperature=0.7, top_p=0.9, deadline=None):
    """Greedy (seeds=None) or actual-logit sampling, with one seed per question.

    No top-k, repetition penalty, forced prefix or generation-config defaults.
    Fold each sequence seed by its token index, independently of batching.
    """
    questions = [questions] if isinstance(questions, str) else list(questions)
    if batch_size < 1 or max_new_tokens < 0 or not 1 <= length <= 256:
        raise ValueError("Invalid generation batch, token cap, or sequence length")
    if seeds is not None and (len(seeds) != len(questions) or temperature <= 0 or not 0 < top_p <= 1):
        raise ValueError("Sampling needs one seed per question, temperature > 0, and 0 < top_p <= 1")
    results = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        prompts = [chat_ids(tokenizer, question) for question in batch]
        if any(len(prompt) + max_new_tokens > length for prompt in prompts):
            raise ValueError("A prompt plus continuation exceeds the generation length")
        prompts += [prompts[-1]] * (batch_size - len(prompts))
        ids = np.full((batch_size, length), tokenizer.pad_token_id, np.int32)
        mask = np.zeros_like(ids)
        for row, prompt in enumerate(prompts):
            ids[row, :len(prompt)], mask[row, :len(prompt)] = prompt, 1
        active = np.arange(batch_size) < len(batch)
        generated = [[] for _ in prompts]
        if seeds is not None:
            batch_seeds = list(seeds[start:start + len(batch)]) + [0] * (batch_size - len(batch))
            keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(batch_seeds, dtype=jnp.uint32))
        for token_index in range(max_new_tokens):
            if deadline is not None and perf_counter() >= deadline:
                break
            positions = np.maximum(mask.cumsum(-1, dtype=np.int32) - 1, 0)
            next_positions = mask.sum(-1, dtype=np.int32)
            logits = editor["full"](theta, editor["fixed"], ids, mask, positions,
                                     (next_positions - 1)[:, None])
            if seeds is None:
                tokens = np.asarray(jnp.argmax(logits[:, 0], axis=-1))
            else:
                draw_keys = jax.vmap(lambda key: jax.random.fold_in(key, token_index))(keys)
                tokens = np.asarray(_sample_tokens(logits[:, 0], draw_keys, temperature, top_p))
            for row in np.flatnonzero(active):
                token = int(tokens[row])
                generated[row].append(token)
                ids[row, next_positions[row]], mask[row, next_positions[row]] = token, 1
                active[row] = token not in STOP_IDS
            if not active.any():
                break
        for answer in generated[:len(batch)]:
            terminated = bool(answer and answer[-1] in STOP_IDS)
            results.append(dict(ids=answer, terminated=terminated,
                                time_capped=bool(not terminated and deadline is not None
                                                 and perf_counter() >= deadline),
                                text=tokenizer.decode(answer, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)))
    return results


def greedy(editor, tokenizer, theta, questions, max_new_tokens=24, batch_size=4, length=128,
           *, deadline=None):
    """Ordinary autoregressive decoding: no supplied answers or output substitutions."""
    return generate(editor, tokenizer, theta, questions, max_new_tokens, batch_size, length,
                    deadline=deadline)


def show_answers(questions, generations):
    """Display complete, unedited generated text with explicit truncation markers."""
    from html import escape

    from IPython.display import HTML, display

    headings = ["Question", *generations]
    rows = ["<tr>" + "".join(f"<th>{escape(h)}</th>" for h in headings) + "</tr>"]
    for i, question in enumerate(questions):
        cells = [escape(question)]
        for answers in generations.values():
            answer = answers[i]
            cells.append(escape(answer["text"]) + ("" if answer["terminated"] else " [token cap]"))
        rows.append("<tr>" + "".join(f'<td style="white-space:pre-wrap;vertical-align:top">{c}</td>' for c in cells) + "</tr>")
    display(HTML("<table>" + "".join(rows) + "</table>"))


def compare(editor, tokenizer, theta, questions, max_new_tokens=64):
    """Show ordinary greedy answers with the original demo's decoding settings."""
    generations = {}
    for label, weights in [("Original", editor["theta0"]), ("NUOX", theta)]:
        generations[label] = greedy(editor, tokenizer, weights, questions,
                                    max_new_tokens=max_new_tokens, batch_size=2, length=256)
    show_answers(questions, generations)
    return generations


def plot_history(history):
    """The task loss and constraint residual are measured at the same weights."""
    import matplotlib.pyplot as plt

    values = np.asarray(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), constrained_layout=True)
    for ax, column, title in zip(axes, (1, 2), ("Edit loss (NLL)", "Max |constraint|")):
        ax.plot(values[:, 0], values[:, column], color="#4263eb", linewidth=2)
        ax.set(xlabel="Update", ylabel=title)
        if column == 1:
            ax.set_yscale("log")
        ax.grid(alpha=0.2)
    plt.show()


DATASET = "Salesforce/wikitext"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATA_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nuox" / "llm_editing"
FILTER_POLICY = "NFKC, casefold, collapsed whitespace; exclude subject substrings"


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def reference_texts(n_texts=128, cache=CACHE, *, exclude_subjects=()):
    """First fixed-order prose rows, excluding the edit's supplied subject names.

    Matching is conservative substring matching after Unicode/whitespace
    normalization. Supply names from the edit specification, never test prompts.
    """
    import pyarrow.parquet as pq

    if n_texts < 1:
        raise ValueError("n_texts must be positive")
    excluded = tuple(normalized_text(name) for name in exclude_subjects)
    if any(not name for name in excluded):
        raise ValueError("Excluded subject names must be nonempty")
    directory = Path(cache) / "reference" / DATA_REVISION
    path = directory / DATA_FILE
    if not path.is_file():
        path = Path(hf_hub_download(
            DATASET, DATA_FILE, repo_type="dataset", revision=DATA_REVISION,
            local_dir=directory, etag_timeout=15,
        ))
    texts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["text"]):
        for text in batch.column(0).to_pylist():
            text = text.strip()
            if (len(text.split()) >= 8 and not text.startswith("=")
                    and not any(name in normalized_text(text) for name in excluded)):
                texts.append(text)
                if len(texts) == n_texts:
                    return texts
    raise ValueError(f"Only {len(texts)} qualifying reference texts in the pinned corpus")


def add_moment(total, activations, mask):
    """Accumulate sum(k k.T), retaining the mean and excluding all padding."""
    keys = np.asarray(activations, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if keys.ndim != 2 or keys.shape[1] != total.shape[0]:
        raise ValueError("Activation and moment dimensions disagree")
    if not np.isfinite(keys).all():
        raise FloatingPointError("Nonfinite reference activations")
    total += keys.T @ keys
    return len(keys)


def covariance(editor, tokenizer, *, n_texts=128, length=64, batch_size=4, cache=CACHE,
               exclude_subjects=()):
    """Return C = sum(k k.T)/actual_token_count and cache/provenance metadata.

    Activations come exclusively from theta0, not any edited parameter vector.
    Accumulation is streamed in float64 on the host; the returned objective uses
    float32. No centering, eigenspace selection or ridge is applied here.
    """
    started = perf_counter()
    if length < 1 or batch_size < 1:
        raise ValueError("length and batch_size must be positive")
    excluded = sorted(set(normalized_text(name) for name in exclude_subjects))
    texts = reference_texts(n_texts, cache, exclude_subjects=excluded)
    tokenizer_state = json.loads(tokenizer.backend_tokenizer.to_str())
    # Fast tokenizers mutate these per-call settings; our preprocessing is explicit below.
    tokenizer_state.pop("padding", None)
    tokenizer_state.pop("truncation", None)
    tokenizer_hash = hashlib.sha256(json.dumps(tokenizer_state, sort_keys=True).encode()).hexdigest()
    settings = dict(
        schema=2, model=MODEL, model_revision=editor["revision"], layer=int(editor["layer"]),
        input_dimension=int(editor["shape"][0]), dataset=DATASET,
        dataset_revision=DATA_REVISION, split="train", n_texts=n_texts,
        sequence_length=length, batch_size=batch_size,
        selection="first rows with >=8 words, excluding headings",
        subject_filter_policy=FILTER_POLICY, excluded_subjects=excluded,
        formatting="plain text; no special tokens; right padding; truncate at sequence length",
        text_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        tokenizer_class=type(tokenizer).__name__, tokenizer_size=len(tokenizer),
        tokenizer_sha256=tokenizer_hash,
        padding_token_id=tokenizer.pad_token_id,
        truncation_side=tokenizer.truncation_side,
        transformers_version=version("transformers"), tokenizers_version=version("tokenizers"),
        easydel_version=version("easydel"), jax_version=version("jax"),
        activation_precision="float32", accumulation_precision="float64", centering=False,
    )
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
    path = Path(cache) / f"moment_{fingerprint}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            moment = saved["moment"]
        if metadata["settings"] != settings or moment.shape != (editor["shape"][0],) * 2:
            raise ValueError("Reference moment cache does not match its settings")
        if metadata["token_count"] < 1 or not np.isfinite(moment).all():
            raise ValueError("Reference moment cache has invalid values")
        metadata.update(cache_hit=True, elapsed_seconds=perf_counter() - started)
        return moment, metadata

    dimension = editor["shape"][0]
    total, count = np.zeros((dimension, dimension), np.float64), 0
    # Right-padding is explicit so this routine is unaffected by generation settings.
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], add_special_tokens=False,
                          truncation=True, max_length=length)["input_ids"]
        ids = np.full((batch_size, length), tokenizer.pad_token_id, np.int32)
        mask = np.zeros_like(ids)
        for row, tokens in enumerate(batch):
            ids[row, :len(tokens)], mask[row, :len(tokens)] = tokens, 1
        positions = np.maximum(np.cumsum(mask, axis=1, dtype=np.int32) - 1, 0)
        keys = editor["activations"](editor["theta0"], editor["fixed"], ids, mask, positions)
        count += add_moment(total, keys, mask)
    if not count:
        raise ValueError("Reference corpus has no non-padding tokens")
    moment = (total / count).astype(np.float32)
    metadata = dict(settings=settings, token_count=count, cache_hit=False,
                    elapsed_seconds=perf_counter() - started)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, moment=moment, metadata=json.dumps(metadata, sort_keys=True))
    temporary.replace(path)
    return moment, metadata
