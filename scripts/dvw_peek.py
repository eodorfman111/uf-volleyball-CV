import argparse
import collections
import re


def read_sections(path):
    text = open(path, encoding="latin-1").read()
    sections, name = collections.OrderedDict(), None
    for line in text.splitlines():
        m = re.match(r"^\[(.+)\]\s*$", line.strip())
        if m:
            name = m.group(1)
            sections[name] = []
        elif name is not None:
            sections[name].append(line)
    return sections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dvw")
    p.add_argument("--lines", type=int, default=12)
    a = p.parse_args()
    sec = read_sections(a.dvw)
    print("sections:", ", ".join(f"{k}({len(v)})" for k, v in sec.items()))
    for name in ("3MATCH", "3SET"):
        for l in [x for x in sec.get(name, []) if x.strip()][:8]:
            print(name, ";".join(l.split(";")[:6]))
    scout = sec.get("3SCOUT", [])
    scout = [l for l in scout if l.strip()]
    skills = collections.Counter(l[3:4] for l in scout if len(l) > 3 and l[0] in "*a")
    print("scout lines:", len(scout), "skills:", dict(skills))
    for l in scout[: a.lines]:
        parts = l.split(";")
        print(len(parts), "fields:", ";".join(f"{i}={v}" for i, v in enumerate(parts) if v and i < 14))


if __name__ == "__main__":
    main()
