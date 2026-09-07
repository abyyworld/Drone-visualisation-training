# Safety model

This document is the reasoning behind five invariants that shape the whole
system. They are not disclaimers added at the end. They decide what fields
exist in the wire protocol, what the tablet is allowed to draw, what is
written to the incident log, and what the interface is permitted to say.

`station/core/safety.py` encodes them as matchable patterns, and
`tests/test_safety_invariants.py` walks the repository -- station, PWA, docs --
and fails the build if the banned phrasing appears in operator-facing text.
This file is one of three places where those phrases are allowed to occur,
because it is the file that explains what they are and why they are banned.

---

## The argument in one paragraph

RGB fire detection fails toward **silence**. Thin smoke against a bright sky,
smouldering with no flame, fire under canopy, fire at night: in every one of
those the model returns an empty list. That empty list is byte-identical to
the one it returns for a genuinely empty field. A human operator knows when
they are struggling to see -- glare, haze, a bad angle, the sun behind the
ridge -- and adjusts, or says so. The model has no equivalent signal. It
cannot report that it is struggling, because from the inside, struggling and
seeing nothing look the same. So the interface must never let an empty result
be read as an informed all-clear. Absence of evidence is displayed as absence
of evidence: nothing at all.

Everything below follows from that.

---

## Invariant 1 -- the absence of a detection is never rendered as a negative claim

No "all clear". No "0 fires detected". No green status light that reads as
safety. No "nothing detected". No "area is clear".

**Why the phrasing matters more than it looks.** A negative claim is a
*finding*: it says an observation was made and it came back negative. An empty
detection list is not a finding, it is a null result, and the difference is
the entire safety case. Rendering the second as the first converts "the model
produced nothing on this frame" into "there is no fire here", which the system
has no basis whatsoever to assert.

**Why a human will make that conversion if you let them.** Under load, people
read interfaces for the answer, not for the epistemics. An incident commander
looking at a screen for two seconds between radio calls will take a green
panel as reassurance no matter what the tooltip says. The only reliable
defence is to not draw the panel. This is why the invariant is enforced by a
build-breaking test rather than by a code-review convention: the pressure to
add a reassuring indicator is constant, it always arrives with a good reason
attached, and it will eventually win an argument if the argument is allowed to
happen every time.

**How it shows up in the code.**

- There is no field in the protocol that asserts absence. `FrameDetections`
  has a `detections` list and no `all_clear`, no `is_safe`, no `fire_count`.
  See the module docstring of `station/core/types.py`.
- `FrameDetections.max_conf` returns `None` when there are no detections,
  not `0.0`. Zero is a number, and numbers get rendered as gauges, and a gauge
  reading zero looks like a measurement of nothing-there.
- The tablet draws boxes when there are boxes and draws nothing when there are
  none. There is no empty state, because an empty state is a claim.

## Invariant 2 -- the overlay augments video someone is already watching

The system says *look here*. It never says *there is nothing there*.

That premise is what makes the rest of the design safe. The operator's primary
instrument is the video. The overlay adds attention-direction on top of it: it
can point at something the operator has not noticed yet, and that is worth a
lot on a 1080p feed of a smoky ridge. What it cannot do is subtract -- the
operator's own scan of the picture is never replaced, reduced or deferred by
the overlay's silence.

The direct consequence, and it is a load-bearing one: **degrading the overlay
to nothing is a safe state.** When the pipeline stalls, when the data channel
drops, when synchronisation fails, the right move is to stop drawing and say
so. The tool falls back to plain video, which is what the operator was using
anyway. That is why `docs/CONTRACT.md` can afford to make the staleness rule
absolute -- there is always a working fallback underneath it.

Compare with the alternative architecture, burning boxes into the video
pixels. There the overlay cannot be withdrawn: a failed or stale detection
becomes part of the picture, indistinguishable from the picture, and the
operator has no way to see that the annotation layer has died. That is the
main reason detections travel as JSON on a data channel in this design, over
and above the bandwidth and the multi-tablet argument.

## Invariant 3 -- no automated alerting anyone acts on without seeing the video

No SMS. No siren. No push notification that says a fire has been found at a
location. Nothing that causes a person to commit a resource on the strength of
the model alone.

**Why.** An alert that travels beyond the screen loses the video with it.
Whoever receives the alert cannot see what the box was drawn around, and the
model's precision on aerial fire imagery is not remotely good enough to spend
a crew on unseen. Worse, an alerting channel silently inverts invariant 1: if
alerts arrive when there is fire, then the absence of an alert starts being
read as the absence of fire -- and now the null result is being broadcast as a
negative finding to people who cannot even check.

**What is allowed.** Drawing attention *on the screen of the person already
watching that screen*. That is not alerting, it is highlighting, and the human
verifies it in the same glance.

**If alerting is added later** -- and someone will ask -- the minimum bar is
that the alert carries the frame, the recipient is a trained operator, the
recipient confirms visually before anything is dispatched, and the silence of
the channel is never presented as information. Design that in the open, in
this document, before writing the code.

## Invariant 4 -- everything is logged, including the empty frames

Every inferred frame goes into `incidents/<id>/detections.jsonl`: timestamp,
media pts, model name, version, weights hash, confidence threshold, and the
detections, which are frequently an empty list.

**Why the empty frames are the important ones.** A log that recorded only
frames with detections would be a highlight reel of what the model happened to
catch. The measurement that decides whether anyone may rely on this system is
the *false-negative* rate (`docs/VALIDATION.md`), and that measurement needs
exactly the frames where something was there and the model returned nothing.
Those frames only exist in the log if empty results are written faithfully.

**Why the model identity is logged with every frame.** Six months later,
reviewing an incident where a plume was missed, the first question is which
weights were running. `ModelInfo.weights_sha` answers it even when someone
forgot to bump `model_version` -- and someone will forget.

**Handling.** Incident logs contain footage of real emergencies and the places
they happened. `.gitignore` excludes `incidents/`. Treat a log as an
operational record that may become evidence: agree retention and access with
the department before the first real incident, not after.

## Invariant 5 -- it is a situational-awareness aid, never a "detector"

`station/core/safety.py` defines `PRODUCT_DESCRIPTOR = "situational-awareness
aid"` and the pattern list bans `fire detector` and `smoke detector` in
operator-facing text.

**Why the word matters.** "Detector" is a term of art with an installed
meaning: a device that has been type-tested against a standard and that a
building's life-safety plan is allowed to depend on. Calling this a detector
imports that expectation wholesale -- of certification, of a known miss rate,
of a duty to alarm -- none of which this system has.

### EN 54 and UL 268 do not apply, and that is exactly the point

Those are the standards people reach for when they hear "fire detection":

- **EN 54** -- fire detection and fire alarm systems: control panels, point
  and line smoke detectors, heat detectors, sounders, and, in EN 54-29/-30/-31,
  multi-sensor units. Written for **fixed installations in buildings**.
- **UL 268** -- smoke detectors for fire alarm systems, the North American
  counterpart, likewise for **fixed installations**, with prescribed fire test
  scenarios in a defined room geometry.

Neither covers a camera on a drone, and neither covers an overlay on a video
feed. There is no certification scheme this system could pass and no
conformity mark it could carry. The naive reading of that is "no standard
applies, so nothing constrains us". The correct reading is the opposite: the
constraints those standards would have imposed -- a specified sensitivity, a
tested miss rate, a defined environment, a maintained installation -- are
absent, so **the framing has to carry the whole load by itself**.

That is the connection between the standards paragraph and the vocabulary
rule. A certified detector may be relied on because a test regime bounded its
failures. This system has no such bound, therefore it must never be described
in a way that invites reliance. It is an aid to a human who is looking, and it
is only ever as good as the looking.

Two practical corollaries:

- Nothing here should be sold, donated or handed over with language implying
  compliance, approval or certification. If a department's procurement asks
  which standard it meets, the honest answer is "none applies; here is the
  validation we did ourselves", and `docs/VALIDATION.md` is that document.
- The claim being made is narrow and defensible: *on footage like this, in
  conditions like these, this model highlighted N of M plumes a human could
  see, and missed the rest in the following ways.* That is a claim the project
  can actually support.

---

## What `station/core/safety.py` enforces

Each pattern carries the reason it is banned, and the reason is printed in the
test failure. That is deliberate: a contributor deleting a pattern should have
to argue with the argument, not with a regex.

| Banned pattern | Why |
|---|---|
| `all clear` | asserts the scene is safe; the model cannot know that |
| `no fire/smoke/detections detected/found/present/visible` | reports a negative finding as a finding |
| `0 fires detected` | a zero count reads as a measurement; it is only a null result |
| `area/zone/scene/site is clear/safe/secure` | states the area is safe -- the exact claim this system must never make |
| `nothing detected/found/to report` | phrases a null result as a reassuring report |
| `safe to enter/proceed/approach` | an operational instruction with no basis |
| `status: ok/good/normal/clear/safe` | a green light that reads as "no fire", not "pipeline healthy" |
| `fire detector` / `smoke detector` | implies a certified device |
| `guarantees detection/safety/coverage` | no such guarantee exists |

`ALLOWED_CONTEXTS` lists the three files where these strings may legitimately
appear: the invariant's own definition, the test that enforces it, and this
document.

### The test

`tests/test_safety_invariants.py` reads every Python, JavaScript, HTML, CSS,
Markdown and YAML file in the repository, skips the allowed contexts, and
fails with the file, the line number, the matched text and the reason.

It is a text search, and text searches are shallow -- it cannot catch a green
status pill with no text on it, an icon that reads as a tick, or a layout that
implies coverage. Those need a human reviewer. What the test does reliably is
prevent the *slow* failure: the gradual accretion of reassuring copy, added a
phrase at a time by people each of whom had a reasonable local motivation, in
a codebase where nobody is left who remembers why the rule existed. That is
the realistic way this system would become dangerous, and it is worth a build
step.

### Reviewing a UI change against the invariants

Ask, in order:

1. Does anything on screen change appearance when the detection list is
   empty, other than boxes disappearing? If yes, that is a negative claim in
   visual form and the test will not catch it.
2. Could a stale or stalled pipeline look identical to a quiet scene? If yes,
   the liveness reporting has a hole -- see `PipelineStatus` and the staleness
   rule in `docs/CONTRACT.md`.
3. Does any text describe the scene rather than the pipeline? Status text is
   about the machine: connected, running, degraded, stalled, overlay stale,
   overlay unsynchronised. None of those are statements about the world.
4. Would a tired person reading this for two seconds take away a reassurance
   the system cannot support? That is the only test that matters, and it is
   the one only a person can run.

---

## Known failure modes, stated plainly

These belong in the operator briefing, not only in a repository:

- **Thin smoke on a bright sky.** Low contrast, and the class the model is
  weakest on. Expect misses.
- **Smouldering with no visible flame.** Little or no colour signature in RGB.
  Expect misses.
- **Fire under canopy.** Often no line of sight at all. The system cannot see
  what the camera cannot see.
- **Night.** An RGB sensor at night is a different imaging problem from the
  daytime footage the model was trained on. Treat night performance as
  unvalidated until it is measured.
- **Small or distant fire.** Below roughly 20 px on the model's input, recall
  falls off hard. Flying lower or raising `inference.imgsz` helps; both cost
  something.
- **Motion blur and rapid panning.** Degrades every class.
- **False positives that look convincing.** Sun glint on water or metal roofs,
  dust plumes behind vehicles, low cloud and fog banks, chimney and industrial
  smoke, orange-brown ground. These are why a human confirms every box.

The design response to that list is not to promise it will improve. It is that
the operator is watching the video, the overlay only ever adds a hint, and the
system never says the words that would let a miss become a decision.

## The `person` class

Person detection was added at the department's request and with approval. It
changes the risk profile of the system, so it is worth being explicit about
what changed and what did not.

**What did not change** is the invariant. The overlay says *look here*. It has
never said *there is nothing there*, and it says it no less loudly now that the
thing it might miss is a person.

**What changed** is how much the invariant matters. A flame front is bright,
large and persistent. A person seen from a drone at altitude is a handful of
pixels, is routinely hidden by canopy, smoke or terrain, holds still, is often
lying rather than standing, and looks a great deal like a rock. Misses are not
an edge case for this class - over occupied ground, an empty screen is the
expected output.

That is tolerable in a tool nobody reads as a search. It is catastrophic in one
somebody does. So four things are structural rather than advisory:

1. **No count, ever.** Not "3 people", not "0 people". A count implies you know
   the denominator; you know only the boxes drawn. `station/core/safety.py`
   fails the build on counting language.
2. **No accountability claim.** "All personnel accounted for", "sector
   evacuated", "building is empty", "search complete" - all refused. Personnel
   accountability comes from roll call and crew tracking. A camera cannot do it
   and must not appear to.
3. **Never anonymised on screen.** `app/js/overlay.js` gives person its own
   colour, dash pattern, corner treatment, printed word and a minimum drawn
   size. A class with no rule falls through to a thin white box labelled `?`,
   and a human being is not a `?`. `tests/test_person_class.py` enforces this.
4. **A looser temporal window.** Person confirms at 2-of-6 against the global
   3-of-5, because a person flickers more than a flame front and a track that
   never confirms is a person never drawn. This buys a busier overlay, which is
   the right way round for this class.

**The failure that actually worries us** is not a false box. It is a crew
watching a screen with no boxes on it, over ground where someone is lying under
canopy, and reading that screen as information. Nothing in the software can
prevent that if the system is described wrongly - which is why it is described,
everywhere, as a situational-awareness aid over video an operator is already
watching, and never as a search tool.

Validation follows from that: `docs/VALIDATION.md` requires false negatives to
be measured for person separately, and broken out by occlusion and posture,
because a model that finds standing people in the open and nobody under trees
will look excellent in aggregate and fail at the only moment it matters.

