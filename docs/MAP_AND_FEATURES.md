# Feature and map contract

All features are receiver-observable proxies. None is ground-truth attack identity.

## Movement: six inputs

| Feature | Meaning | Fixed divisor |
|---|---|---:|
| dx, dy | Change in claimed position from previous received message with same pseudonym | 10 m each |
| speed | Reported speed | 40 m/s |
| acceleration | Reported acceleration | 10 m/s² |
| heading_sin, heading_cos | Circular representation of reported heading | 1 each |

Coordinates are not used as absolute movement inputs. A smooth constant offset can therefore be difficult for movement alone to detect. Context/map checks are intended to test that gap, not guaranteed to solve it.

## Communication timing: seven inputs

| Feature | Meaning | Fixed divisor |
|---|---|---:|
| receive_gap | Current minus previous reception time | 1 s |
| send_gap | Current minus previous claimed sending time | 1 s |
| delay | Receive time minus claimed send time | 0.1 s |
| sender_rate_1s | Count of this pseudonym's received messages in the inclusive last second | 10 |
| receive_jitter | Population std of the last up to 10 receive gaps | 0.2 s |
| repeat_id_fraction | (message count − unique IDs) / message count, last second | 1 |
| repeat_payload_fraction | Same repetition measure for position/speed/acceleration/heading tuples | 1 |

Negative send gaps and suspicious delays are retained. Clock synchronization is assumed for interpreting receive-minus-send delay; deployed clock offset would need correction. Count-in-window is a rate proxy, not exact offered load. Loss and radio power are not invented from missing fields. Message-ID jumps are not assumed to equal packet loss. Repeated stationary payloads can be legitimate, so repetition is evidence rather than an automatic attack label.

## Neighbour/optional road context: ten inputs

| Feature | Meaning | Fixed divisor |
|---|---|---:|
| neighbor_count | Fresh, already-observed other pseudonyms within configured radius | 20 |
| neighbor_median_speed | Median speed of those neighbours | 40 m/s |
| speed_difference | Sender speed minus neighbour median | 40 m/s |
| nearest_distance | Closest such neighbour to claimed sender position | 300 m |
| heading_disagreement | Median smallest angular difference | 180° |
| receiver_distance | Distance between sender claim and receiver position | 300 m |
| road_distance | Distance to nearest external-map centreline segment | 10 m |
| road_heading_difference | Smallest difference from matched segment direction | 180° |
| has_neighbors | Availability flag | 1 |
| has_road_map | Availability flag | 1 |

Unavailable neighbour-dependent values and unavailable road values are 0 with an explicit mask. These are not measurements of “zero congestion” or “exactly on the road”. Neighbour count is affected by radio visibility, pseudonym changes and Sybil behaviour. Pseudonyms are not assumed to identify unique physical vehicles reliably.

No raw simulator `distance_to_road_edge`, `driversProfile`, `sender_id` or attack metadata is used by the networks. The `*_noise` fields are not added to the released reported values a second time; exact sensor-error interpretation should be checked against the dataset release you use. This prototype consumes the released position/speed/heading as reported measurements and records that assumption.

## Optional map format

Create a JSON map using centreline vertices in the **same local metre projection** as the dataset. For example (illustrative coordinates only):

```json
{
  "coordinate_system": "local_meters",
  "roads": [
    {"points": [[1000, 2000], [1100, 2000]], "bidirectional": false},
    {"points": [[1000, 2100], [1100, 2100]], "bidirectional": true}
  ]
}
```

Do not use that illustration for real VeReMi coordinates. Convert the matching SUMO network lane geometry to the same coordinate system, then verify sampled message positions visually. Segment order defines permitted travel direction when `bidirectional` is false. Heading convention is 0° north and 90° east.

The supplied matcher chooses the nearest segment by distance. It is a baseline, not robust lane-level map matching: parallel lanes, junctions, road widths, bridges and projection errors can create ambiguity. Off-centre distance is not itself proof of invalid driving. Validate map-aware and neighbour-only experiments separately.

## Canonical records

`messages.csv.gz` retains labels and provenance for auditing, but `models.py` consumes only fixed-dimensional arrays. Timestamps are explicitly converted from manifest units to seconds. Arrays are float32. Features are not clipped merely because they are anomalous; non-finite values cause an error.

Training windows use ten prior movement states, ten timing states including the current message, and current context. Any attack label in the prior/current window disqualifies a fitting window. The primary protocol already limits fitting to clean baseline archives.
