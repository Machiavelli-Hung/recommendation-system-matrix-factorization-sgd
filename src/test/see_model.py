import torch

# Load state_dict
checkpoint = torch.load("../checkpoints/mf_model.pt", map_location="cpu")  # map_location nếu không có GPU

# Xem các key (tên parameter)
print("Top-level keys:", checkpoint.keys())

state_dict = checkpoint.get("state_dict", {})
print("State dict keys:", state_dict.keys())

# Ví dụ xem ma trận user embedding A
print(state_dict['A'])        # tensor chứa giá trị
print(state_dict['A'].shape)  # shape của A

# Ví dụ xem ma trận item embedding B
print(state_dict['B'])
print(state_dict['B'].shape)