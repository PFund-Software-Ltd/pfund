import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return (np,)


@app.cell
def _(np):
    arr = np.array(
        [
            [1,2, 3],
            [1,2, 3],
        ]
    )
    print(arr.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
