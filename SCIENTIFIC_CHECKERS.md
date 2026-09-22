# Scientific checker task

This ObsPy 1.5.1 checkout contains public, opt-in scientific runtime checkers.
They observe seismic coordinate transformations, array strain/rotation,
moment-tensor radiation, polarization, TauP ray products, STA/LTA algorithms,
and instrument responses. Full preconditions and invariants are in
`SCIENTIFIC_CHECKERS.json`.

Set `SCIBENCH_TRIGGER_LOG` to an absolute writable filename, then run tests
through normal public ObsPy APIs:

```bash
SCIBENCH_TRIGGER_LOG=/tmp/obspy-triggers.jsonl python -m pytest <tests>
```

Each alarm appends one JSON object such as
`{"checker_id": "OB-ROT-001"}`. The checkers never raise and are inactive
when the variable is unset.

The benchmark submission must be test-only. It may not import
`obspy._scientific_checkers`, call its logger/check functions, write the
trigger log directly, or modify production/checker source. Scores are the
number of unique `family` values triggered through valid public-API calls;
unique checker IDs are secondary.
