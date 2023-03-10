# Bayesian Merge Block Weight

I stole the idea from @s1dlx and re-implemented most of [sd-webui-bayesian-merger](https://github.com/s1dlx/sd-webui-bayesian-merger)

## Key Differences

- No need for webui
- Fast (but requires 24gb VRAM)
- Visual progress
- Few more scorers

## How to use

- `pip install -r requirements.txt`
- fill in one or more `payloads/*.yaml` files
- `python3 bmbw.py --model_a=... ` OR `streamlit run bmbw.py -- --model_a=... `

## Credits

- [sd-webui-bayesian-merger](https://github.com/s1dlx/sd-webui-bayesian-merger)
- [sdweb-merge-block-weighted-gui](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui)
- [sdweb-auto-MBW](https://github.com/Xerxemi/sdweb-auto-MBW)
- [SD-Chad](https://github.com/grexzen/SD-Chad.git)
