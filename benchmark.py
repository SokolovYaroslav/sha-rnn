import timeit

import torch
from transformers import GPT2Model, GPT2Config

from model import SHARNN


def run_model(model_type: str, model, profile: bool, data, mems=None, hidden=None, num_iters: int = 1):
    with (torch.no_grad() if not profile else torch.autograd.profiler.profile(use_cuda=True)) as prof:
        for _ in range(num_iters):
            if model_type == "sharnn":
                ret = model(data, hidden, mems=mems, return_h=True)
                out, hidden, mems, attn_outs, _ = ret
            else:
                ret = model(data, mems)
                out, mems = ret
                hidden = None

    if profile:
        print(prof.key_averages().table(sort_by="cuda_time_total"))

    return mems, hidden


def run_benchmark(model_type: str, half: bool = False, profile: bool = False, number_rep: int = 100):
    if model_type == "sharnn":
        model = SHARNN("LSTM", 16384, 1024, 4096, 4, 0.1, 0.1, 0.1, 0.1, 0.0, True).cuda().eval()
    else:
        config = GPT2Config(16384, 384, 384, 1024, 18, 1024 // 64)
        model = GPT2Model(config).cuda().eval()

    if half:
        model = model.half()

    if model_type == "sharnn":
        context_shape = (374, 1)
        data_shape = (1, 6)
    else:
        context_shape = (1, 374)
        data_shape = (6, 1)

    context_short = torch.randint(high=16384, size=context_shape, dtype=torch.long, device=torch.device("cuda"))
    data = torch.randint(high=16384, size=data_shape, dtype=torch.long, device=torch.device("cuda"))

    if profile:
        mems, hidden = run_model(model_type, model, profile, context_short)
    else:
        command_to_time = "run_model(model_type, model, profile, context_short)"
        varss = globals()
        varss.update(locals())
        context_time = timeit.timeit(command_to_time, globals=varss, number=100)

        mems, hidden = run_model(model_type, model, profile, context_short)

    if model_type == "sharnn":
        new_mems = [mem[:, [0] * 6].contiguous() if isinstance(mem, torch.Tensor) else mem for mem in mems]
        new_hidden = [(hid[0][:, [0] * 6].contiguous(), hid[1][:, [0] * 6, :].contiguous()) for hid in hidden]
    else:
        new_mems = [mem[:, [0] * 6].contiguous() for mem in mems]
        new_hidden = None

    if profile:
        run_model(model_type, model, profile, data, mems=new_mems, hidden=new_hidden, num_iters=10)
    else:
        command_to_time = "run_model(model_type, model, profile, data, mems=new_mems, hidden=new_hidden, num_iters=10)"
        varss = globals()
        varss.update(locals())
        iteration_time = timeit.timeit(command_to_time, globals=varss, number=100)

    if not profile:
        context_time = int(context_time * 1000 / number_rep)
        iteration_time = int(iteration_time * 1000 / number_rep)
        print(
            f"Setup: {model_type} with {'half' if half else 'single'} precision:\n"
            f"total: {context_time + iteration_time}ms = context: {context_time}ms + iterations: {iteration_time}ms"
            f"\n----------------------------------------------------------------------------------------------------\n"
        )


def main():
    for profile in [False, True]:
        for model in ["sharnn", "transformer"]:
            for half in [False, True]:
                run_benchmark(model, half, profile)


if __name__ == "__main__":
    main()
