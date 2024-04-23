import timeit

code1 = """
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
A = torch.rand(5096,5096, dtype=torch.float16, device = device )

mask = torch.full((A.shape[-1], A.shape[-1]), torch.finfo(A.dtype).min, device=device)
mask_cond = torch.arange(mask.size(-1), device=device)
mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0).to(device)
"""


code2 = """
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
input_emb = torch.rand(5096,5096, dtype=torch.float16,device = device)
mask = torch.full(
            (1, 1,input_emb.shape[-1], input_emb.shape[-1]), 1, device=device
        )
mask = torch.log(torch.triu(mask, diagonal=0)).to(input_emb.dtype)
"""

# Đo thời gian thực thi cho phương pháp 1
time_taken1 = timeit.timeit(stmt=code1, number=1000)
print("Exec time of 1st method:", time_taken1)

# Đo thời gian thực thi cho phương pháp 2
time_taken2 = timeit.timeit(stmt=code2, number=1000)
print("Exec time of 2nd method:", time_taken2)
