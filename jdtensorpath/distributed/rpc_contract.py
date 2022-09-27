import torch

def rpc_contract(contraction_list, arrays):
    for contraction in contraction_list[:-1]:#last one is transpose for final result correction
        order_operand, do_einsum, einsum_str_or_axes = contraction
        temp_operands = [arrays.pop(x) for x in order_operand]
        if do_einsum:
            #print(einsum_str_or_axes)
            #print(temp_operands)
            result = torch.einsum(einsum_str_or_axes[0], *temp_operands)
        else:
            #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape)
            #print(temp_operands)
            result = torch.tensordot(*temp_operands, dims=einsum_str_or_axes[0])
        arrays.append(result)
    final_transpose = contraction_list[-1]
    result = arrays[0].permute(final_transpose)
    return result

def rpc_contract_GPU(contraction_list, list_arrays):

    gpus_count = int(torch.cuda.device_count())
    if not gpus_count:
        raise ValueError("There's no GPU in this computer! please use cpu mode!")

    if len(list_arrays) > gpus_count:
        raise ValueError("Number of GPUs is smaller than number of assigned slices!!")

    available_gpus = [torch.device('cuda:'+''.join(str(i))) for i in range(gpus_count)]  # pylint: disable=no-member
    host_device = list_arrays[0][0].device

    results = []
    for i in range(len(list_arrays)):
        GPU_arrays = [array.to(available_gpus[which_gpu]) for array in list_arrays[i]]
        tmpt = rpc_contract(contraction_list, GPU_arrays)
        tmpt = tmpt.to(host_device)
        results.append(tmpt)

    result = sum(results)

    return result