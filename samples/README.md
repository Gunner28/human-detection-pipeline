# Samples

| File | Committed | What it is |
|---|---|---|
| `manbenz.png` | yes, 112 KB | One man, one car. The image that exposed the duplicate-box defect. |
| `Samplevideo2.mp4` | **no**, 17 MB | 2.3 min street footage used during development |

`manbenz.png` is worth keeping because it is the regression case: the raw
model reports **three** people and two cars on it. After suppression:
one person, two cars.

```bash
python cli.py image samples/manbenz.png                    # 1 person
python cli.py --raw --all-classes image samples/manbenz.png  # 3 people, 2 cars
```

## Using your own footage

Point the CLI at any video:

```bash
python cli.py count /path/to/your_video.mp4
```

**What works best:** a fixed camera on a doorway or corridor. A tripod
removes camera motion, a doorway makes the counting line correspond to
something physical, and walking through a known number of times gives you
ground truth for free.

**What works badly:** the development clip. It is an edited montage of
unrelated scenes — a blurred close-up, a wide high street, a ground-level
shot of legs — with handheld motion throughout. Fine for proving the
pipeline runs; useless for measuring accuracy, since tracking assumes
continuity and there is none.
