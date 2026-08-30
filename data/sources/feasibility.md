# Comment-source feasibility audit

Audit date: 2026-08-29. This is a local preparation artifact, not an instruction for the GPU-side
Agent. Platform and language are not eligibility constraints: any language may support any target
character. The hard requirements are target relevance, auditable provenance, irreversible author
hashing, and a stable collection path.

| Source | Browser result | Evidence fields | Volume signal | Decision |
|---|---|---|---|---|
| YouTube watch pages | Stable after scrolling; public watch-page comments also reproducibly exposed through yt-dlp | comment ID, public author ID, relative time, full text, video ID and canonical URL | Large analysis videos available for English and Chinese roles | Selected primary source. Capture on the connected preparation host, keep raw comments private/non-redistributable, and locally filter target relevance. |
| Bilibili video pages | Stable in Chrome; the browser-facing signed WBI comment pages reproduce public roots and available replies | numeric user ID, timestamp, full text, video ID, capture completeness | Character-analysis videos with hundreds to thousands of comments | Selected primary source. Record nested-page failures and never claim a partial capture is complete; locally filter target relevance. |
| Stack Exchange | Stable public search and official API; CC BY-SA attribution path | post/comment ID, user ID, timestamp, body, canonical URL | Strong for Harry Potter; role-dependent elsewhere | Preferred auditable supplement. Use the official API rather than HTML collection. |
| Fandom Discussions | Stable public post search | post URL, public author identity, post/reply text | Multiple role-specific discussions found for Hermione; role-dependent | Supplement after confirming the applicable community/license terms. |
| Baidu Tieba | Stable after user login; virtual scrolling advanced the inspected thread from floors 18 to 34 to 53 | thread/floor context, stable account ID in profile URL, date, full floor text | Sample thread has 177 replies; large work-specific bars exist | Strong browser-only supplement. Capture each loaded window before scrolling because older floors are virtualized out of the DOM. |
| Hupu | Public post body and replies stable in Chrome | thread URL, account profile ID, date, reply text | Sample `甄嬛传` post had 15 replies; search exposes more threads | Browser-visible supplementary source only: normal navigation and visible replies, no bulk crawler or hidden endpoint. Preserve every thread URL. |
| PDB | Directory visible, but profile/comment rendering repeatedly timed out | partial snippets and profile URLs | High apparent vote/discussion volume | Not a primary source because extraction is unstable and discussion paths are restricted. |
| Douban | Direct subject/review navigation redirected to a security page | none collected | Unknown | Not usable in the current browser state. Do not bypass the security page. |
| Zhihu | Search page repeatedly timed out | none collected | Unknown | Not a primary source. |
| MovieChat | Cloudflare challenge page | none collected | Unknown | Excluded; do not bypass the challenge. |
| Reddit | Not tested for bulk extraction after policy audit | n/a | Likely high | Excluded from automated collection because `robots.txt` disallows all crawling. |

## Selected source mix

- The frozen corpus uses Bilibili and YouTube broadly, with Stack Exchange official API, Fandom
  Discussions, signed-in Tieba, and visible Hupu threads where suitable. Platform and language are
  not tied to an experimental panel.
- Harry Potter: add Stack Exchange through its official API and Fandom Discussions through public browser pages.
- Hupu is a supplementary browser source, not a bulk-crawl source. No robots bypass, hidden API, or authentication workaround is allowed.
- A comment is accepted only if it discusses the target's behavior, motives, values, appraisals,
  affect/coping, relationships, self-narrative, situation-response pattern, or expressive style.
  Merely appearing under a target-focused video/thread is not sufficient.
- Preserve original text and `language`. Translation is allowed as a derived analysis artifact but is
  not required by the multilingual retriever/LLM and must never erase the source text or provenance.
- Exact accepted counts, source threads, author coverage, collection methods, import records, capture
  completeness, and the final corpus SHA-256 are frozen in `data/audits/corpus_inventory.json`.
