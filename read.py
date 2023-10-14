from  dataloader import *

train_dataset = DeepFashion2COCODataset("data/validation/image" , "data/validation_json.json", transforms=data_transforms)

val_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
for i, (x, y) in enumerate(val_loader):
	if i <= 5:
		print(x.size())
		x_x = x[0].numpy().transpose(1,2,0)
		print(x_x.shape)
  
		print(y.size())
		y_y = y[0].numpy().squeeze(0)
		print(y_y.shape, np.max(y_y))
		
		# 스케일링을 해주어야 합니다. 원래의 텐서가 -1~1 사이였다면,
		y_y = ((y_y * 0.5 + 0.5) * 255).astype(np.uint8) *30
		x_x = ((x_x * 0.5 + 0.5) * 255).astype(np.uint8)
		# y_y = y_y.astype(np.uint8)
		# x_x = x_x.astype(np.uint8)
		# NumPy 배열을 PIL Image 객체로 변환하고 파일로 저장합니다.
		Image.fromarray(y_y).save(f'output_y{i}.png')
		Image.fromarray(x_x).save(f'output_x{i}.png')
	else:
		break