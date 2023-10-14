from dataloader import *
import time

def train(model, config):
    """
    """
    # dataset_train = DeepFashion2Dataset()
    # dataset_train.load_coco(config.train_img_dir, config.train_json_path)
    # dataset_train.prepare()
 
    # dataset_valid = DeepFashion2Dataset()
    # dataset_valid.load_coco(config.valid_img_dir, config.valid_json_path)
    # dataset_valid.prepare()
 
    # model.train(dataset_train, dataset_valid,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=30,
    #             layers='3+')
    
    checkpoint_dir = "model"
    
    train_dataset = DeepFashion2COCODataset(config.train_img_dir , config.train_json_path, transforms=data_transforms)
    
    val_dataset = DeepFashion2COCODataset(config.valid_img_dir , config.valid_json_path, transforms=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCH = 5
    
    model.to(device)
    for epoch in range(NUM_EPOCH):
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            if batch <= 1:
                print(x.size())
                x_x = x[0].numpy().transpose(1,2,0)

                y_y = y[0].numpy().squeeze(0)

                y_y = ((y_y * 0.5 + 0.5) * 255).astype(np.uint8)
                x_x = ((x_x * 0.5 + 0.5) * 255).astype(np.uint8)

                Image.fromarray(y_y).save(f'output_y{batch}.png')
                Image.fromarray(x_x).save(f'output_x{batch}.png')
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output['out'], y.squeeze(1).long())
            loss.backward()
            optimizer.step()
            if batch % 200 == 0:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{current_time}] [{epoch+1}/{NUM_EPOCH}][{batch}/{len(train_loader)}] Loss: {loss.item()}")

        checkpoint_path = os.path.join(checkpoint_dir, f"deepfashion2_fcn_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint at epoch {epoch+1} to {checkpoint_path}")
        

    # 훈련 세트 이외의 데이터를 사용하여 모델을 평가합니다.
        model.eval()

        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # 사용자 정의 평가 지표를 계산할 수도 있습니다.

        with torch.no_grad():
            for batch, (x_val, y_val) in enumerate(val_loader):  # 검증 DataLoader 사용
                
                if batch <= 1:

                    x_x = x_val[0].numpy().transpose(1,2,0)

                    y_y = y_val[0].numpy().squeeze(0)

                    y_y = ((y_y * 0.5 + 0.5) * 255).astype(np.uint8)*30
                    x_x = ((x_x * 0.5 + 0.5) * 255).astype(np.uint8)

                    Image.fromarray(y_y).save(f'val_y{batch}.png')
                    Image.fromarray(x_x).save(f'val_x{batch}.png')
                x_val, y_val = x_val.to(device), y_val.to(device)

                # 모델 예측
                output = model(x_val)
                
                # 손실 계산
                loss = criterion(output['out'], y_val.squeeze(1).long())
                val_loss += loss.item()
                if batch <= 1:
                    o = output['out'][0]
                    print(output['out'].size())
                    masks = torch.argmax(o, dim=0).squeeze(0).cpu().numpy()
                    print(masks.shape)
                    masks = ((masks * 0.5 + 0.5) * 255).astype(np.uint8) *30
                    Image.fromarray(masks).save(f'val_mask{batch}.png')
                # 정확도 계산
                _, predicted = torch.max(output['out'], 1)

                correct_predictions += (predicted == y_val.squeeze(1)).sum().item()
                total_samples += y_val.size(0)

        # 검증 데이터셋에 대한 손실과 정확도 출력
        average_val_loss = val_loss / len(val_loader)
        accuracy = correct_predictions / total_samples

        print(f"Validation Loss: {average_val_loss} Validation Accuracy: {accuracy * 100}%")


    