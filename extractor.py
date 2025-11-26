import json

with open("BosonPlusSim.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

with open("output.py", "w", encoding="utf-8") as out_file:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            out_file.write("# ---\n")
            out_file.write("".join(cell.get("source", [])))
            out_file.write("\n\n")