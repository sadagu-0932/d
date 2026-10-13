"""solutions/pNN.txt -> docs/data/pNN.json, docs/data/index.json

Source format:
  section: 01 함수의 극한      (optional, page heading)
  concept                      (page with no problems)
  @ 01 | answer-in-latex       (starts a problem)
  latex line                   (one solution step, rendered as display math)
  " plain text line            (one solution step, rendered as text)
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "solutions"
OUT = ROOT / "docs" / "data"


def parse(path):
    page = {"section": "", "problems": []}
    concept = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("section:"):
            page["section"] = line[len("section:"):].strip()
        elif line == "concept":
            concept = True
        elif line.startswith("@"):
            no, _, ans = line[1:].partition("|")
            page["problems"].append({"no": no.strip(), "answer": ans.strip(), "steps": []})
        elif line.startswith('"'):
            page["problems"][-1]["steps"].append({"text": line[1:].strip()})
        else:
            page["problems"][-1]["steps"].append({"tex": line})
    return page, concept


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = {}
    for path in sorted(SRC.glob("p*.txt")):
        num = int(re.fullmatch(r"p(\d+)\.txt", path.name).group(1))
        page, concept = parse(path)
        if concept:
            pages[num] = {"problems": 0}
            continue
        pages[num] = {"problems": len(page["problems"]), "section": page["section"]}
        (OUT / f"p{num}.json").write_text(
            json.dumps(page, ensure_ascii=False, indent=1), encoding="utf-8")
    index = {"min": min(pages), "max": max(pages), "pages": pages}
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {len(pages)} pages ({index['min']}-{index['max']})")


if __name__ == "__main__":
    main()
