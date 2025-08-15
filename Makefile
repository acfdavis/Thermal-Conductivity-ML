train-tf:
	python scripts/train_tf.py

    train-torch:
	python scripts/train_torch.py

    predict-tf:
	python scripts/predict_from_csv_tf.py --input data/example_input.csv --output data/predictions_tf.csv --model models_tf/tf_model.keras --scaler models_tf/scaler.joblib

    predict-torch:
	python scripts/predict_from_csv_torch.py --input data/example_input.csv --output data/predictions_torch.csv --model models_torch/torch_model.pt --scaler models_torch/scaler.joblib

    up:
	docker compose up --build

    curl-tf:
	curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"formulas":["SiO2","GaN","Bi2Te3"]}'

    curl-torch:
	curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d '{"formulas":["SiO2","GaN","Bi2Te3"]}'
