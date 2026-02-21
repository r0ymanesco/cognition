# Plan

This is a conversation I had with Claude about this idea. The following is the transcript.

### Me
What are some of the issues in agent retrieval strategies that use grep or code to explore large document stores?

### Claude
Great question. Using grep or code-based search to explore large document stores in agentic retrieval pipelines has some real limitations:

**Scalability and Performance**

Grep is linear — it scans every file byte by byte, which becomes painfully slow on large corpora. Even with parallelism, it doesn't scale the way indexed search does. Large document stores can have millions of files spanning terabytes, making brute-force scanning impractical within the latency budgets of an interactive agent.

**Exact Match Brittleness**

Grep matches literal strings or regular expressions. If the query says "vehicle" but the document says "automobile," you miss it. Agents relying on grep can't handle synonyms, paraphrasing, or semantic variation without increasingly complex regex patterns that become hard to maintain and still fall short of semantic search.

**Context Blindness**

Grep returns lines or snippets, not coherent passages. The agent gets a fragment with little surrounding context, which makes it harder to determine relevance or extract a meaningful answer. Rank-ordering results by relevance is also non-trivial — grep just returns matches in file order.

**No Ranking or Relevance Scoring**

Code-based search lacks built-in relevance ranking. Without TF-IDF, BM25, or vector similarity, the agent has no way to prioritize the most relevant hits from among potentially thousands of matches, forcing it to either process everything (expensive) or truncate arbitrarily (lossy).

**Structural Ignorance**

Grep treats documents as flat text. It doesn't understand document structure — headings, sections, tables, metadata — so it can't restrict search to, say, "the conclusion section" or "rows where the date is after 2020" without brittle custom parsing.

**Multi-hop and Compositional Queries**

When an agent needs to find information that requires combining facts across documents, grep requires multiple passes with manually chained queries. There's no native support for joins, document graph traversal, or compositional retrieval.

**Chunking and Boundary Problems**

Code-based approaches often chunk documents naively (e.g., fixed token windows), which can split concepts across chunks. A relevant answer might straddle a chunk boundary and never appear in any single retrieved unit, causing the agent to miss it entirely.

**Index Staleness and Overhead**

If you build indexes to compensate (e.g., inverted indexes), you now have to maintain them as documents change. Keeping indexes fresh at scale is an engineering burden, and a stale index can cause the agent to miss new documents or retrieve deleted content.

**Security and Access Control**

A grep-style agent scanning a filesystem may inadvertently surface documents it shouldn't, bypassing document-level access controls that a proper retrieval system would enforce at query time.

The general direction most production systems move toward is hybrid retrieval — combining sparse lexical search (BM25) with dense vector/semantic search, layered on top of proper document chunking strategies, metadata filtering, and re-ranking. This gives agents both the precision of keyword matching and the flexibility of semantic understanding.

### Me
This recent paper called "Recursive Language Model" basically uses code as a form of compression on large context. What are some of the downsides to this paper's strategy?

### Claude
Let me search for that paper so I can give you accurate, specific feedback rather than speculation.Good, I now have a clear picture of the paper. Here are some meaningful downsides to the RLM strategy:

**Cost Explosion from Recursive Sub-Calls**

The paper itself notes an example where Qwen3-Coder defined a function that uses a sub-LM call *per line*, leading to thousands of recursive sub-calls on a full input — at a total cost of $0.27 for a single query. That might sound small, but at scale across many queries, the economics become brutal. The model has no reliable way to bound how many sub-calls it will spawn, making cost unpredictable.

**The Decomposability Assumption**

A key argument for RLMs is that most complex tasks can be decomposed into smaller, "local" sub-tasks. But this assumption breaks down for tasks requiring holistic understanding — things like detecting subtle thematic contradictions across a document, assessing stylistic coherence, or reasoning about aggregate patterns that only emerge from global context. Local sub-tasks can miss the forest for the trees.

**Information Loss at Aggregation**

Even though the paper claims to avoid the information loss of summarization, the aggregation step where sub-LM results get combined and passed back to the root model is itself a form of lossy compression. The root model only sees the *outputs* of sub-calls, not the full content those sub-calls processed. Nuance, uncertainty, and edge cases in the sub-documents can disappear at this boundary.

**Verbatim Copy and Output-Heavy Tasks**

For tasks like verbatim copy, performance suffers because of the increased number of tool-calls and turns involved. When the task inherently requires the model to reproduce or closely engage with large chunks of text rather than reason about it, the code-as-compression framing actively fights against the task structure.

**Recursion Depth is Currently Limited**

The paper only considers a recursive depth of 1 — the root LM can call LMs, but not other RLMs. This is a real constraint. For truly hierarchical reasoning over deeply nested or multi-layered document structures, a single level of recursion may be insufficient, and deeper recursion multiplies all the cost and complexity issues.

**Dependency on Code Generation Quality**

The whole system's reliability rests on the root model writing correct, well-structured code to navigate the context. If the model generates subtly wrong parsing logic, mishandles edge cases in document formatting, or uses inefficient traversal strategies, there's no graceful fallback — errors propagate through the recursive call tree silently or expensively.

**Benchmarks May Not Capture the Hard Cases**

The evaluation focuses on tasks like needle-in-a-haystack and OOLONG, which are well-structured and amenable to programmatic decomposition. Real-world long-document tasks — legal contracts with cross-references, scientific papers with implicit dependencies, conversational transcripts with context-dependent meaning — are messier and may resist the clean chunking the RLM relies on.

**Training Data Scarcity**

The paper itself acknowledges that the optimal mechanism for implementing RLMs remains under-explored, and expresses excitement about future work that explicitly trains models to reason as RLMs. Right now, the approach leans heavily on frontier models that weren't trained for this paradigm — it's prompting a behavior rather than building it in. Smaller or open models may struggle significantly, and the post-trained RLM-Qwen3-8B they release is only a preliminary step.

**Sandboxing and Security Surface**

Running a live Python REPL that can make arbitrary LM sub-calls introduces a real security surface. A malicious or adversarial document could potentially influence the code the root model writes — a form of prompt injection that could trigger unintended sub-calls, exfiltrate data through API calls, or consume resources by causing the model to spawn runaway recursion.

### Me
This paper also treats code as a form of compression of long contexts. What do you think of that?

### Claude
It's a genuinely interesting and somewhat novel framing, but I think it's worth interrogating carefully — because it's partly right and partly a useful metaphor that can obscure some real limitations.

**Where the framing holds up**

Code really does compress in a meaningful sense. Instead of passing 10 million tokens of raw text through a model's context window, you pass a short Python snippet that *describes how to process* that text. The instructions for navigating the data are vastly smaller than the data itself — which is exactly what compression is. There's a clean analogy to run-length encoding or other classical schemes, where you store a transformation rule rather than the full output.

The framing also captures something true about *semantic* compression. A function like `classify_by_sentiment(chunk)` encodes a rich semantic operation in a few tokens. That's a form of compression that summarization can't match, because summarization collapses information while the function defers it — the information is still there, waiting to be retrieved.

**Where it gets philosophically slippery**

Traditional compression is lossless or has precisely characterized loss. Code-as-compression here is neither — its "decompression" step (running sub-LM calls) is itself stochastic and error-prone. The model at the bottom of the recursion might hallucinate, misread its chunk, or fail to extract the relevant signal. So unlike a zip file, you can't decompress back to the original faithfully. The compression ratio is real, but the fidelity guarantee isn't.

There's also a subtle conflation happening between *compression of representation* and *compression of attention*. What the RLM is really doing is preventing the root model's attention mechanism from having to span the full context — which is a systems-level win, not purely an information-theoretic one. The information still has to be processed somewhere; it just gets shunted to sub-calls. Calling that compression risks making it sound like the information burden disappears, when it's actually redistributed.

**The deeper issue**

Classical compression assumes the compressor has a global view of the data before deciding how to encode it. The RLM's root model doesn't — it probes the context partially, writes code based on what it can infer from a peek, and then commits to a traversal strategy. That's more like *adaptive sampling* than compression. Sometimes the model chooses a great strategy; sometimes, as we saw with the Qwen3-Coder example spawning thousands of sub-calls, it chooses a catastrophically inefficient one because it misjudged the data's structure.

This is actually where the connection back to your first question about grep is interesting. Grep-style retrieval has the same structural problem: the query strategy is chosen before the content is fully understood. The RLM improves on grep by making the strategy itself dynamic and semantic, but it inherits the same fundamental blindness — you're writing a compression scheme without having seen the full data you're compressing.

**What the framing is most useful for**

As a *design intuition*, code-as-compression is valuable. It pushes you toward thinking about what minimal representation of an operation you need to pass between model calls, rather than naively shuttling large text blocks around. That's a genuinely productive way to think about agentic architecture. But as a *formal claim* about what the system achieves, it should be held loosely — because the guarantees that make compression useful in classical CS (reversibility, bounded loss, predictable ratios) don't straightforwardly carry over to a system where the decompressor is a probabilistic language model.

### Me
You're right, I think it's not strictly compression but rather a representation that's more efficient. IMO it doesn't get around the problem of a finite context window as the "hist" can still grow beyond the model effective context window. What I'm interested in is a mechanism or scaffolding around LLMs that can at least theoretically make the context window infinite by making even the "hist" explorable or offloaded elsewhere. Not sure if there is a logical inconsistency with what I'm missing though


### Claude
This is a really sharp observation, and you're identifying something real. The RLM paper does essentially kick the can down the road — it offloads the *input* context but the execution history, intermediate variables, and accumulated turns still live in a context window that can rot or overflow. You haven't solved the fundamental problem, you've just deferred it.

**What you're gesturing toward**

What you want is something like: an agent whose *working state* — not just its inputs, but its own reasoning trace, intermediate outputs, and task history — is itself stored externally and queryable on demand. So at any point in time, the context window holds only: current query + locally relevant retrieved state + enough scaffolding to make the next decision. The total task length becomes theoretically unbounded while the active context stays bounded.

This is actually the architecture of a **Turing machine**. The finite state machine (the LLM, with bounded context) reads and writes to an infinite external tape (memory/history store). The LM is just the transition function; all state lives outside it. So the theoretical framework for "infinite context" already exists in CS — it just requires that the finite part be expressive enough to navigate the infinite part correctly.

**The logical tension you're sensing**

I think the inconsistency is real and deep, and it comes down to this: **offloading the hist only works if the task has a Markov-like property with respect to some compact state representation.** That is, at any point in the task, a bounded working context must be *sufficient* to make the next correct decision — the relevant past must be retrievable and synthesizable in bounded space.

The problem is that many interesting tasks don't have this property. They have **irreducible long-range dependencies**:

- Writing a long novel where the ending must be consistent with dozens of subtle setups from the beginning
- Multi-hop reasoning where a chain of inference builds on itself in ways that can't be compressed into a summary without loss
- A software project where architectural decisions made 10,000 lines ago constrain what you can do now in non-obvious ways

For these tasks, "retrieve the relevant history" is circular — you can only know what's relevant *after* you've understood the global structure, but understanding the global structure is exactly what requires holding more than fits in the window.

**The retrieval oracle problem**

There's a secondary issue: to query the hist intelligently, the agent needs a model of *what's in the hist*. That model either lives in the current context (context pressure again) or is itself stored externally (leading to an infinite regress of indexes and meta-indexes). At some level you always need a finite "directory" that fits in the window to navigate the infinite external store — and that directory is itself a form of lossy compression of history.

**What might actually work**

The most theoretically coherent version of what you're describing would require:

1. **Structured, addressable external memory** — not a flat hist log but something with semantic keys, so the agent can retrieve by content rather than scanning linearly
2. **A compact task state object** — a minimal representation of "where we are" that's small enough to always fit in the window, sufficient for the agent to resume correctly, and updated incrementally rather than grown unboundedly
3. **A task decomposition guarantee** — the overall task architecture must ensure that each local decision is indeed separable from sufficiently distant history, so that "retrieve relevant snippets" is actually well-posed

The honest conclusion is that this works *in theory* for any computable task (by the Turing machine argument), but the practical challenge is that LLMs aren't great finite state machines — they need more working context than a classical FSM to do useful reasoning, and the "retrieve the right thing" step is itself an inference problem that can fail. So you get a system that's theoretically unbounded but practically fragile, especially on tasks with dense global dependencies.

The version of this that feels most promising is probably an LLM that's explicitly trained to maintain a compact, structured task-state object and update it incrementally — essentially learning to be its own state machine rather than relying on the raw context window as implicit state.

### Me
A few questions
1. You're right that this is only consistent if there is some markov like property to the task. But it's also true that even with a lossy representation of history, as long as it's dynamic with respect to the current step, it should work reasonably apart from very degenerate cases. Would you agree?
2. Your last suggestion about an LLM trained to maintain a compact state that's updated incrementally hints at an RNN. That's been shown to be hard to scale, at least it didn't beat transformers are much more finite.

### Claude
Both of these are sharp pushbacks and I think you're right on both counts.

**On the first point**

Yes, I'd agree. I was being too strict by implying it only works with a true Markov property. The weaker condition you're describing — that a lossy but *contextually dynamic* representation of history is sufficient — is much more reasonable, and honestly is how humans function. We don't have lossless recall of everything we've ever done, but we have a retrieval system that's biased toward what's relevant right now, and that works well enough for almost all practical tasks.

The degenerate cases where it breaks down are real but narrow: tasks where a specific detail from arbitrarily far back is both critical and unpredictable in advance — so no dynamic retrieval strategy would have known to preserve it. In practice, most long-horizon tasks don't have that property at every step. The failure mode is real but shouldn't be the primary objection to the architecture.

**On the second point**

This is a really good catch, and it cuts deep. What I was describing — a model that maintains a compact hidden state and updates it incrementally at each step — is structurally exactly an RNN. And the history of RNNs is instructive: they're theoretically Turing-complete and can represent infinite context through their hidden state, but in practice they fail because:

- Gradients vanish or explode over long sequences during training, making it hard to learn dependencies that span many steps
- The fixed-size hidden state becomes an information bottleneck that the model can't expand on demand
- They're hard to parallelize during training, which limits scale

Transformers "won" precisely by abandoning incremental state in favor of full attention over the whole sequence — trading memory efficiency for trainability and parallelism. So there's a real irony here: the architectural property that makes transformers so trainable (attending to everything) is exactly what creates the context window problem you're trying to solve.

What's interesting though is that this might be a false dichotomy going forward. A few threads worth considering:

The external memory approach — if the "state" isn't a fixed-size vector baked into weights but a structured, readable external store, you sidestep the information bottleneck problem that killed RNNs. The model doesn't have to compress history into a hidden state; it writes structured artifacts that it can read back selectively. This is more like a von Neumann architecture than an RNN.

There's also recent work like Mamba and other state-space models trying to get RNN-like efficiency while being more trainable — but so far they haven't clearly beaten transformers on hard reasoning tasks, which suggests the fundamental tension is still unresolved.

So the honest position is probably: the RNN analogy is correct and it's a warning sign, but the reason RNNs failed was specific to how their state was represented and trained — and an external structured memory system might avoid those failure modes while preserving the theoretical benefit. Whether that actually scales in practice is still an open question.

### Me
Regarding the point about RNNs, I agree, and it's why I'm exploring inference time scaffolds around current architectures based on transformers that might actually achieve the promise of an RNN without the same limitations. My first point is an attempt at trying to make sense of this approach. Can we explore that a bit more and try to nail down some actionable strategies for experimentation?

### Claude
This is a great framing for an experimental research program. Let me try to be as concrete as possible.

**The core abstraction to nail down first**

The RNN analogy is actually a useful guide here, because it tells you exactly what components you need to replicate at the scaffold level:

- A **state object** (the hidden state analog) that persists across steps
- A **read operation** (what gets pulled into context at each step)
- A **write operation** (how state gets updated after each step)
- A **transition function** (the LLM itself)

The key difference from an RNN is that the state doesn't have to be a fixed-size vector — it can be a richly structured, variable-size external store. That's your escape hatch from the RNN bottleneck problem. So the design space is really about how you architect those three operations.

**Actionable experimental dimensions**

*State representation*

The first thing to pin down is what the state object actually looks like. A few candidates worth experimenting with:

A flat append log with a retrieval index is the simplest baseline — essentially a vector database of past reasoning steps. Easy to implement, but retrieval quality depends heavily on embedding quality and the assumption that semantic similarity correlates with relevance, which isn't always true.

A structured schema-based object is more interesting — something like a typed dictionary where the model explicitly writes to named fields ("current hypothesis," "open questions," "established facts," "discarded paths"). This forces the model to be deliberate about what it's preserving and makes retrieval more deterministic. The risk is that a fixed schema may not generalize across task types.

A hierarchical summary tree is a middle ground — recent history stored verbatim, older history progressively compressed into summaries at increasing levels of abstraction. This mirrors how human working memory actually operates and is consistent with your point about lossy-but-dynamic being sufficient.

*The read operation*

This is probably where the most interesting experimentation lives, because it's the place your first point directly applies. The question is what makes the retrieval "dynamic with respect to the current step." A few strategies:

Query-based retrieval where the model explicitly generates a retrieval query before accessing state. The model says "what do I need to know right now?" before reading — this makes the retrieval adaptive rather than static.

Attention-weighted recency where more recent state is always included and older state is retrieved by relevance. The intuition is that recency is a cheap proxy for relevance in most tasks.

Step-type-conditioned retrieval — if you can classify the current step (e.g., planning vs. execution vs. verification), you can use different retrieval strategies for each. Planning steps might need broad historical context; execution steps might only need local recent state.

*The write operation*

This is underexplored relative to retrieval. The naive approach is to just append every step's output to the store. But smarter strategies include:

Differential writes — only write what's genuinely new, not what's already represented in state. This requires the model to compare its current output against existing state, which is expensive but keeps the store from bloating.

Confidence-weighted writes — high-confidence conclusions get written as facts; low-confidence ones get written as hypotheses with explicit flags. This makes the state self-annotating about its own reliability.

Invalidation — when the model reaches a conclusion that contradicts something in state, it explicitly marks prior entries as superseded. This is the hardest to get right but most important for long-horizon correctness.

**The experiment I'd start with**

The cleanest minimal experiment would be: take a task that definitively requires more context than a model's window — something like multi-hop QA over a large document set, or a long code refactoring task — and implement the simplest possible scaffold: a structured state object with explicit read/write steps, query-based retrieval, and flat append with no invalidation.

Then ablate each component systematically:
- Fixed retrieval vs. query-based retrieval (tests your first point directly — does dynamic retrieval matter?)
- Flat log vs. schema-based state (tests whether structure helps the model write better)
- No compression vs. hierarchical compression of old state (tests the lossy history hypothesis)

The metric you care about isn't just final accuracy — it's also state efficiency: how much of the stored state actually gets retrieved and used? A high-quality scaffold should show that retrieval is sparse but precise, which would validate the Markov-like sufficiency argument empirically.

**The deeper question for the research program**

The thing worth keeping in mind is that this scaffold essentially asks the LLM to do something it wasn't explicitly trained to do — manage its own external state coherently over long horizons. So a natural next question after initial experiments is whether you can fine-tune or prompt-train the model to be a better state manager, without full retraining. That's where this becomes a genuinely novel training paradigm rather than just a clever inference wrapper.

### Me
One thing I want to add here is that any ablation should show that this new theoretically infinite context window scaffold is significantly better than what exists today: namely RLM. Another thing is, this sounds interesting, but I'm not entirely sure it would actually be significantly better than RLM. RLM can supposedly achieve 10M+ token context already

### Claude
This is exactly the right skepticism to have before investing in an experimental program. Let me try to reason through whether there's a genuine gap worth pursuing.

**Where RLM is actually strong**

The 10M+ token result is real and impressive, but it's worth being precise about *what kind of tasks* it excels at. RLM works well when:

- The task can be decomposed into chunks that can be processed independently
- The answer can be assembled from locally extractable signals
- The document structure is regular enough for code to navigate it

For those tasks, RLM is probably close to a ceiling already. So if your proposed scaffold does roughly the same thing, you won't beat it meaningfully.

**Where RLM has structural weaknesses your approach could exploit**

The honest case for a better scaffold has to be grounded in tasks where RLM's architecture is fundamentally inadequate, not just inefficient. A few candidates:

*Tasks with state that evolves over time.* RLM has no persistent state across queries — each call starts fresh. If you're doing a long-horizon task like iterative code development, research synthesis, or multi-session planning, RLM has no memory of what it decided two steps ago unless you manually pass it in. A scaffold with explicit persistent state is architecturally suited to this in a way RLM isn't.

*Tasks requiring invalidation or belief revision.* RLM processes chunks and aggregates results, but it has no mechanism for saying "the conclusion I drew from chunk 3 is invalidated by what I found in chunk 47." The root model sees outputs, not a coherent evolving world model. A scaffold that explicitly tracks hypotheses and contradictions could handle this class of task far better.

*Tasks where the query itself evolves.* RLM essentially fixes its decomposition strategy early and executes it. But for open-ended research or exploratory reasoning, what you're looking for changes as you learn more. A scaffold where the read operation is dynamically conditioned on current state can adapt its retrieval strategy mid-task in a way RLM can't.

**The honest concern though**

Even granting those gaps, the empirical question is whether they show up on benchmarks that exist today. Most current long-context benchmarks — OOLONG, BrowseComp, needle-in-a-haystack — are precisely the retrieval-and-aggregation tasks where RLM shines. To demonstrate your scaffold's superiority you'd likely need to either:

- Find or construct benchmarks specifically targeting stateful, multi-session, or belief-revision tasks — which is a non-trivial research contribution in itself
- Show that on RLM's own benchmarks your scaffold matches it with significantly less compute, i.e. a cost efficiency argument rather than a quality argument

**The real differentiator**

I think the strongest argument for your approach over RLM isn't "handles more tokens" — that's a race you might not win. It's that RLM is fundamentally **stateless and task-local**, while what you're describing is **stateful and task-global**. Those are different things solving different problems. RLM is essentially a better retrieval mechanism. What you're building is closer to a cognitive architecture — something that can reason coherently across sessions, revise its beliefs, and maintain a world model that evolves with the task.

That reframing also gives you a cleaner research narrative: you're not competing with RLM on its own benchmarks, you're opening up a class of tasks that RLM can't address at all. The ablation against RLM then becomes a negative result you want — showing that RLM fails on stateful tasks while your scaffold doesn't — rather than trying to beat it on tasks it was designed for.

Does that distinction feel like a productive direction, or do you want to push on whether stateful tasks are actually important enough to warrant the engineering investment?

### Me
That's valid. Given this, I'd like to make sure the proposed experiments are in fact able to show the distinction between a stateful vs stateless architecture

### Claude
This is the right thing to nail down before building anything. Let me think through what experimental design actually isolates that variable cleanly.

**The core requirement for a discriminating benchmark**

For an experiment to show the stateful vs. stateless distinction, the task must have at least one of these properties:

- **Temporal dependency**: a correct decision at step N requires remembering a specific decision made at step M, where M is far enough back that it falls outside any reasonable context window
- **Belief revision**: information encountered late in the task should change conclusions drawn early, requiring the system to retroactively update its state
- **Cumulative constraint satisfaction**: the task has global constraints that only become visible when integrating information across many steps — no local chunk tells the whole story

RLM can't handle any of these well by construction. The question is whether you can find or build tasks that test them cleanly without introducing confounds.

**Proposed task classes**

*Multi-session iterative refinement*

Give the system a large codebase and ask it to implement a feature across multiple separate sessions, where each session only reveals part of the requirements. A stateless system has no memory between sessions and will make locally reasonable but globally inconsistent decisions. A stateful system should accumulate constraints and flag contradictions.

The discriminating metric isn't just final correctness — it's **consistency of decisions across sessions**. RLM will likely show inconsistency that grows with the number of sessions, while a stateful scaffold should show flat or slowly growing inconsistency.

*Contradiction detection across a long document stream*

Feed the system documents one at a time, where a claim in document 1 is subtly contradicted by document 47, and the correct answer requires noticing that contradiction. RLM processes chunks independently and aggregates — it will likely miss the contradiction or not flag it with appropriate confidence. A stateful system that maintains a hypothesis store with explicit invalidation should catch it.

The key design requirement here is that the contradiction must be **semantic, not lexical** — otherwise RLM's grep-style code could catch it too. Something like: document 1 establishes that a company's policy prohibits X, and document 47 describes an action that implies X was done, without ever using the word "policy" or "prohibit."

*Exploratory research with evolving queries*

Ask the system to investigate an open-ended question where the right sub-questions to ask change as you learn more. For example: "what are the failure modes of system X" where early documents suggest one taxonomy of failures, but later documents reveal a completely different framing that makes the early taxonomy obsolete. 

A stateless RLM will commit to its initial decomposition strategy and miss the reframing. A stateful scaffold whose read operation is conditioned on current state should detect that its working hypothesis needs revision.

**The confound you need to control for**

The biggest risk in these experiments is that RLM could simulate statefulness by including prior outputs in its context on each call — essentially manually passing state through. You need to either:

- Make the accumulated state large enough that passing it all through violates RLM's context budget, forcing it to compress or drop history
- Or measure not just whether RLM gets the right answer, but how it gets there — specifically whether it's maintaining a coherent evolving model or just getting lucky through local reasoning

The second point matters because a well-prompted RLM might occasionally produce correct answers on stateful tasks through coincidence, which would muddy your results. So your evaluation needs to probe the *reasoning trace*, not just the final output.

**Metrics that actually discriminate**

Final accuracy alone won't show the distinction clearly enough. You want:

- **State utilization rate**: what fraction of prior decisions get correctly retrieved and applied at each step? A stateful scaffold should show high, targeted retrieval. RLM should show either zero (no history) or degrading performance as history grows.
- **Contradiction recovery rate**: when contradictory information is introduced, how often does the system revise its prior conclusions correctly? This is a direct test of belief revision capability.
- **Decision consistency across sessions**: for multi-session tasks, measure whether decisions made in session 3 are consistent with constraints established in session 1. Plot this as a function of session distance — you'd expect RLM to degrade linearly while your scaffold holds flat.
- **Cost as a function of task length**: RLM's sub-call costs grow with input size. A well-designed stateful scaffold should show much flatter cost curves on long-horizon tasks because it's not reprocessing history — it's querying a structured store.

**The experimental sequence I'd recommend**

Start with contradiction detection — it's the cleanest to implement, easiest to evaluate objectively, and most obviously impossible for a stateless system. It also lets you tune your write and invalidation operations in isolation before tackling the harder multi-session and evolving-query tasks.

Then move to multi-session refinement, which tests persistence directly and gives you the session-distance consistency metric that most cleanly separates the architectures.

Save the evolving query task for last — it's the most interesting scientifically but also the hardest to evaluate because "the query should have evolved here" is somewhat subjective without careful benchmark design.

Does this feel like a clean enough separation, or do you see ways RLM could still pass these tests that we need to design around?
