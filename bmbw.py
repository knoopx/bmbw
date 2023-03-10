import click
from bmbw.optimizer import Optimizer


@click.command()
@click.option("--model_a", type=click.Path(exists=True), required=True)
@click.option("--model_b", type=click.Path(exists=True), required=True)
@click.option("--device", type=str, default="cuda:0")
@click.option(
    "--config_path",
    type=click.Path(exists=True),
)
@click.option("--init_points", type=int, default=1)
@click.option("--n_iters", type=int, default=10)
def main(*args, **kwargs) -> None:
    bo = Optimizer(*args, **kwargs)
    bo.start_optimization()


if __name__ == "__main__":
    import streamlit as st
    from transformers import logging
    import torch
    import torch.backends
    import torch.backends.cuda
    import warnings

    logging.set_verbosity_error()
    warnings.simplefilter("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True
    st.set_page_config(page_title="BMBW", layout="wide")

    main()
