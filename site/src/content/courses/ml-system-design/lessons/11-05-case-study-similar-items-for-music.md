---
id: "11.05"
course: "ml-system-design"
module: "11"
slug: "case-study-similar-items-for-music"
title: "Case study: similar items for music"
keyIdeas: ["\"Similar tracks\" and \"similar artists\" combine collaborative signals (co-listening in sessions and playlists) with content signals (audio and metadata).", "Co-listening embeddings learned like word embeddings, with playlists or sessions as sentences, capture taste well for popular items.", "Audio embeddings place new and long-tail tracks in the same space, solving cold start for items nobody has played yet.", "Similarity must be filtered and diversified (no duplicates, versions or the same artist repeated) and evaluated with both offline proxies and listening tests."]
question: "Design a \"similar tracks\" feature for a music streaming service with 100 million tracks and 400 million users. It powers a \"fans also like\" list and automatic radio stations seeded from a track. Explain the signals, embeddings, cold start for new tracks, filtering and evaluation."
answer: "Learn track embeddings from co-occurrence in user-made playlists and listening sessions (skip-gram with negative sampling, or matrix factorisation), down-weighting very popular tracks and ignoring skipped plays. Learn audio embeddings with a model trained on spectrograms, aligned to the collaborative space so that new tracks can be placed by audio alone; combine both with weights that shift towards collaborative as play counts grow. Retrieve nearest neighbours with an approximate index, then filter out duplicates, alternate versions and the same artist beyond a cap, and apply licensing and region rules. For radio, add a sequencing step that balances similarity to the seed with diversity and occasional discovery. Evaluate offline with next-track prediction in held-out sessions and playlist completion, plus human listening judgements; online with skip rate, saves and listening time on radio. Monitor coverage of the catalogue and new-release reach. Trade-offs: familiarity versus discovery, and popularity bias versus long-tail exposure."
rubric: ["Combines collaborative (co-listening, playlists) and content (audio, metadata) signals", "Describes how co-listening embeddings are trained and how popularity and skips are handled", "Solves cold start for new tracks with audio embeddings mapped into the shared space", "Uses approximate nearest-neighbour retrieval over the catalogue", "Filters duplicates, versions and repeated artists, and applies licensing rules", "Evaluates with held-out sessions or playlists plus listening tests", "Measures online skip rate, saves and listening time", "Discusses popularity bias and familiarity versus discovery"]
sources: [{"title": "Barkan and Koenigstein, Item2Vec: Neural Item Embedding for Collaborative Filtering (2016)", "url": "https://arxiv.org/abs/1603.04259", "note": "Word2vec-style embeddings for items in sessions."}, {"title": "Spijkervet and Burgoyne, Contrastive Learning of Musical Representations (2021)", "url": "https://arxiv.org/abs/2103.09410", "note": "Self-supervised audio embeddings for music."}, {"title": "Wikipedia, Music information retrieval", "url": "https://en.wikipedia.org/wiki/Music_information_retrieval", "note": "Background on audio features and tasks."}, {"title": "Wikipedia, Collaborative filtering", "url": "https://en.wikipedia.org/wiki/Collaborative_filtering", "note": "Item-based methods."}]
practice: ["word2vec-sgns-epoch", "implicit-als-user-factors", "embedding-similarity", "catalog-coverage-and-novelty"]
tags: ["music similarity", "co-listening embeddings", "audio embeddings", "cold start"]
---

## Requirements

Clarify the uses: a "fans also like" list on track and artist pages, and radio stations that play an endless sequence starting from a seed track. Assume 100 million tracks (most with few plays), 400 million users, tens of thousands of new tracks a day, and radio requests needing the next tracks within 200 ms.

## Metrics

Online: skip rate on radio (especially skips within 30 seconds), saves and playlist adds, listening time per station, and clicks on similar-item lists. Offline: next-track hit rate in held-out sessions, completion of held-out playlists, and human similarity ratings on a sample.

## Signals

- Playlists: user-made playlists are strong evidence that tracks fit together.
- Sessions: tracks played in sequence without skips.
- Audio: spectrogram-based embeddings capture tempo, timbre, instrumentation and mood.
- Metadata: genre tags, release year, label, language, credits.

## Collaborative embeddings

Treat each playlist or session as a sentence and each track as a word, and train skip-gram with negative sampling: tracks that appear near each other get similar vectors. Down-sample very popular tracks, as word2vec down-samples frequent words, so that every track is not similar to the top hits. Exclude plays skipped within a few seconds. Matrix factorisation on the user-track matrix is an alternative with similar results.

## Audio embeddings and cold start

A new track has no plays and therefore no collaborative embedding. Train an audio model (on spectrograms, with self-supervised contrastive learning or supervised on tags), then train a mapping from audio embeddings to collaborative embeddings on tracks that have both. A new track can then be placed by its audio alone on release day. Blend the two as plays accumulate: $e = \alpha\, e_{\text{audio}} + (1-\alpha)\, e_{\text{collab}}$, with $\alpha$ falling as play count rises.

<div class="ex">
<div class="lab">Worked example</div>

**new releases nobody heard**

A service's similar-tracks lists used only co-listening embeddings, so new releases never appeared until they were already popular elsewhere. Small artists' first tracks got almost no radio plays. Adding audio-based placement for tracks with fewer than 1,000 plays, plus a small exploration share on radio, raised first-month plays for new tracks from small artists by about 40% with no change in overall skip rate.

</div>

## Retrieval, filtering and sequencing

Retrieve the top 200 neighbours from an approximate nearest-neighbour index. Then:

- Remove duplicates, live and remastered versions of the seed, karaoke and cover versions (unless wanted).
- Cap tracks per artist.
- Apply licensing and region availability.

For radio, sequence tracks: start close to the seed, then drift gently, mixing familiar tracks (which the user has played) with discoveries, and adapt to skips within the session.

## Serving

~~~dg-flow Similar tracks and radio. Embeddings and neighbours are precomputed; radio sequencing reacts to skips.
Playlists and sessions -(skip-gram training)-> Collaborative embeddings
Audio -(audio model + mapping)-> Audio embeddings
Collaborative embeddings + audio embeddings -(blend)-> ANN index -(top 200)-> *Filter and sequence -(skips update session)-> Radio queue
~~~

## Monitoring

Track skip rate per station type, catalogue coverage (share of tracks ever recommended), the share of plays going to the top 1% of tracks, and new-release reach. Rebuild embeddings weekly and audio placements daily.

## Likely follow-up questions

- "How do you handle a track that appears on many unrelated playlists, like a wedding song?" Weight co-occurrences by playlist coherence and downweight very long or mixed playlists.
- "Similar to what, exactly?" Ask; "similar sound" and "same fans" differ. Audio similarity serves the first, co-listening the second; the blend weight is a product choice.

## Trade-offs

- Familiarity versus discovery: too much discovery raises skips; too little makes radio stale.
- Popularity bias versus long-tail exposure and artist fairness.
- Audio similarity versus taste similarity.

## A numeric sanity check

Embeddings for 100 million tracks at 128 dimensions in 8-bit form take about 13 GB, so the nearest-neighbour index fits on one machine and can be replicated freely. Precomputing the top 200 neighbours for every track is 20 billion pairs, too much to store cheaply for a catalogue where most tracks are rarely seeded; precompute for the few million most-played tracks and query the index live for the rest. A radio station needs only a handful of new tracks every few minutes, so per-request cost is small.

<div class="take">
<div class="lab">Key takeaway</div>

learn track embeddings from co-listening, place new tracks with audio embeddings mapped into the same space, filter versions and repeated artists, and sequence radio to balance familiarity with discovery, judged by skips and saves.

</div>
