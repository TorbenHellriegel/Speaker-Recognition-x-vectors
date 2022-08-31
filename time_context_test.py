import torch

x = torch.tensor([  [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
                    [[4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [1], [2], [3]],
                    [[7], [8], [9], [10], [11], [12], [13], [14], [15], [1], [2], [3], [4], [5], [6]],
                    [[10], [11], [12], [13], [14], [15], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                    [[13], [14], [15], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]])

def get_time_context(x, c=[0]):
    l = len(c) - 1
    xc =   [x[:, c[l]+cc:c[0]+cc, :]
            if cc!=c[l] else
            x[:, c[l]+cc:, :]
            for cc in c]
    return xc

context = [-2, -1, 0, 1, 2]
x_context = get_time_context(x, context)
new_x = torch.cat(x_context, 2)
print(new_x)
print("x:", x.shape)
print("context:", context)
print("new_x:", new_x.shape)

context = [-2, 2]
x_context = get_time_context(x, context)
new_x = torch.cat(x_context, 2)
print(new_x)
print("x:", x.shape)
print("context:", context)
print("new_x:", new_x.shape)

context = [-4, -2, 0, 2, 4]
x_context = get_time_context(x, context)
new_x = torch.cat(x_context, 2)
print(new_x)
print("x:", x.shape)
print("context:", context)
print("new_x:", new_x.shape)

context = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
x_context = get_time_context(x, context)
new_x = torch.cat(x_context, 2)
print(new_x)
print("x:", x.shape)
print("context:", context)
print("new_x:", new_x.shape)

#TODO turn into actual test