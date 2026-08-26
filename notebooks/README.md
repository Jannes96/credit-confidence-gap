# Notebooks

Notebooks in this directory are **exploration only**. Nothing reported in the
README, in `reports/`, or in any figure may originate here.

The rule is not stylistic. A notebook's output depends on the order its cells
happened to be executed in, which is not recorded, not diffable, and not
reproducible from the file alone. Anything whose correctness matters therefore
lives in `src/credit_trust_lab/`, is covered by a test, and is invoked from
`scripts/reproduce_all.py`.

**Workflow:** explore here → once something is worth keeping, move it into
`src/`, write a test for it, and call it from the reproduction script. Then
either delete the notebook cell or leave it as a narrative, knowing it is no
longer the source of anything.

To use the project's environment in Jupyter:

```bash
.venv/bin/pip install jupyterlab && .venv/bin/jupyter lab
```
