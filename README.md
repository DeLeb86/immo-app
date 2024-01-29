# immo-app
This application runs an API with a model trained to predict prices from real estate ads.

## APP

### install and run

Before running the app, install requirements: 

    pip install -r requirements.txt

**Standalone**

Once the needed libraries are installed, run the app as a standalone api : 

    uvicorn app:app --host 0.0.0.0

your app will be available on [http://localhost:8000](http://localhost:8000)

**Docker**

You can also run the app inside a container using docker : 

build image : 

    docker build -t immoapp:latest .

and run container : 

    docker run --network host --name immo_app immoapp:latest
    
### endpoints 

**/predict** : takes multiple args to create the feature vector to be sent to the model : 

    { 
        "PostalCode" : int,
    	 "TypeOfProperty" : int,
    	 "TypeOfSale" : int,
    	 "Kitchen" : Optional[str],
    	 "StateOfBuilding" : Optional[str],
    	 "Bedrooms" : Optional[float],
    	 "SurfaceOfGood" : Optional[float],
    	 "NumberOfFacades" : Optional[float],
    	 "LivingArea" : float,
    	 "GardenArea" : Optional[float]
    }

### training and evaluation

the model behind the app is a `GradientBoostingRegressor`.
The original dataset is to be found in [`resources/workshop_dataset.json`](resources/workshop_dataset.json). The dataset has been splitted in 80% training, 20% for testing. 

Both training and testing sets can be found in :
- [`resources/train.pkl`](resources/train.pkl)
- [`resources/test.pkl`](resources/test.pkl)

those files are serialized dataframes to be loaded with `pickle.load()`.

The model and preprocessor (encoder) has been trained on the train set and evaluated on the unseen test set with a r2 score of 0.78




