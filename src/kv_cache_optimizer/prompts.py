"""
A small, deliberately diverse set of prompts for the multi-prompt benchmark.

Using one prompt (the original version of this project) risks measuring
"how well does this policy handle this one writing style" rather than
"how well does this policy handle eviction in general." Domain and length
are varied on purpose:

  - technical_doc / distributed_systems: structured, high information
    density, the kind of prompt where losing the wrong sentence is obvious
  - narrative: looser prose, tests whether eviction breaks story coherence
    rather than factual coherence
  - qa_instruction: short prompt, instruction-following continuation, tests
    behavior when there's very little protected prefix to lean on
  - dialogue: conversational turn-taking, different attention patterns than
    monologue-style prompts
  - code_explanation: code-adjacent tokens (short, high-entropy identifiers)
    stress the scoring function differently than natural language
  - long_document: the longest prompt, closest to the 4-8k token range the
    project's README says is the actual point of the memory bottleneck

Each entry is (prompt_id, category, text). Keep prompt_id stable across runs
if you add more later, it's used to key paired comparisons in the summary.
"""

from typing import List, Tuple

PROMPTS: List[Tuple[str, str, str]] = [
    (
        "distributed_systems",
        "technical_doc",
        "You are a helpful assistant with expert knowledge of distributed systems. "
        "Below is a long design document. Read it carefully, then continue it with "
        "a detailed section on failure recovery.\n\n"
        "## System Overview\n"
        "The system is a distributed key-value store partitioned across N shards, "
        "each replicated three ways using a Raft-based consensus protocol. Clients "
        "write through a coordinator node, which forwards requests to the shard "
        "leader responsible for the relevant key range. Reads may be served by any "
        "in-sync replica depending on the requested consistency level.\n\n"
        "## Data Model\n"
        "Keys are arbitrary byte strings up to 1KB; values up to 1MB. Each shard "
        "maintains a sorted log-structured merge tree on local disk, with periodic "
        "compaction to bound read amplification. Metadata about shard ownership is "
        "stored in a separate strongly-consistent control plane.\n\n"
        "## Failure Recovery\n"
    ),
    (
        "arctic_expedition",
        "narrative",
        "Continue the following short story in the same tone and point of view.\n\n"
        "The wind had not stopped for three days. Mara checked the sled's lashings "
        "again, more from habit than need, and looked back at the ridge they had "
        "crossed that morning. Somewhere behind it, the rest of the team was still "
        "waiting for a radio call that would not come until the storm broke. She "
        "had food for six more days, fuel for four, and a compass heading that "
        "she was no longer entirely sure of.\n\n"
    ),
    (
        "recipe_substitution",
        "qa_instruction",
        "I'm making a lasagna but I'm out of ricotta cheese. What can I substitute, "
        "and will it change the texture or flavor much? Explain the best options.\n\n"
    ),
    (
        "support_dialogue",
        "dialogue",
        "The following is a chat log between a customer support agent and a user "
        "troubleshooting a failed software installation. Continue the conversation.\n\n"
        "User: The installer gets to 80% and then just closes with no error message.\n"
        "Agent: Thanks for the details. Can you tell me what operating system and "
        "version you're running, and whether you're installing as an administrator?\n"
        "User: Windows 11, and yes I right-clicked and ran as administrator.\n"
        "Agent: Got it. Let's check the installer log file first, it usually points "
        "to the real error even when the UI doesn't show one.\n"
    ),
    (
        "python_cache_explainer",
        "code_explanation",
        "Explain what the following Python function does, then suggest one "
        "improvement.\n\n"
        "def evict(self, cache, keep_indices):\n"
        "    old_len = cache.layers[0].get_seq_length()\n"
        "    for layer in cache.layers:\n"
        "        layer.keys = layer.keys[:, :, keep_indices, :]\n"
        "        layer.values = layer.values[:, :, keep_indices, :]\n"
        "    self.evicted_count += old_len - keep_indices.shape[0]\n"
        "    return cache\n\n"
    ),
    (
        "long_history_essay",
        "long_document",
        "Read the following essay excerpt carefully, then continue it with a "
        "section on how the printing press changed the spread of scientific "
        "ideas in Europe.\n\n"
        "## The Manuscript Era\n"
        "Before the fifteenth century, the reproduction of written knowledge in "
        "Europe depended almost entirely on manual copying, typically performed "
        "by monastic scribes working from existing manuscripts. This process was "
        "slow, expensive, and error-prone: each copy introduced the possibility "
        "of transcription mistakes, and the sheer labor involved meant that only "
        "wealthy institutions, monasteries, universities, and royal courts, could "
        "afford to commission and maintain substantial libraries. A single Bible "
        "could take a scribe the better part of a year to complete. As a result, "
        "the total number of books in circulation across the continent numbered "
        "only in the tens of thousands, and access to any given text was often "
        "restricted to those with direct institutional affiliation.\n\n"
        "## Early Print Culture\n"
        "The introduction of movable type printing in the mid-fifteenth century, "
        "most famously associated with Johannes Gutenberg's press in Mainz, "
        "fundamentally altered this economy of knowledge. What had taken a scribe "
        "months could now be produced in days, and at a fraction of the cost per "
        "copy. Print shops proliferated rapidly across German-speaking lands and "
        "then across the rest of Europe, and by the year 1500 an estimated twenty "
        "million volumes had been printed, a scale of production entirely "
        "unimaginable under the manuscript system. This was not merely a "
        "quantitative shift; it changed who could plausibly own, read, and "
        "annotate a text, and therefore who could participate in written "
        "argument at all.\n\n"
        "## Standardization and Error Correction\n"
        "One underappreciated effect of print was the gradual standardization of "
        "texts that had previously existed in many divergent manuscript "
        "variants, each accumulating its own scribal errors over generations of "
        "copying. Printers could correct an error once and have that correction "
        "propagate across an entire print run, rather than requiring each "
        "individual copy to be checked and fixed by hand. Over time, this made "
        "it far more feasible for scholars in different cities, who had never "
        "met and might never correspond directly, to work from what was "
        "effectively the same text and trust that their citations of it would "
        "be mutually intelligible.\n\n"
    ),
]
