# ML in Policing Example: High-harm prediction using COMPAS
Using COMPAS data in an AutoGluon Fair example to provide data for a worked example of the [RUDI framework](https://www.police-ml.com/introduction) and demonstrate a quick way to develop a baseline algorithm for prediction of high-harm offences for the purposes of suspect prioritisation. 

Many police forces in UK are looking for ways to prioritise suspects based on risks implicitly encoded in their offending histories. Some are turning to ML-based prediction models. It is important to consider ethical aspects of this type of work and to acknowledge limitations of the training data and model performance when attempting to integrate these types of models into the policing workflow. 

This is an example of some of the things one might consider when looking into these types of models, but doesn't cover real issues that might affect a particular organisation. The bulk of ethical work is done at the conceptualisation and rationale stages of planning where you need to examine if a prioritisation algorithm is right for your organisation and if you have means to maintain and evaluate its performance on ongoing basis. Likewise an algorithm will give imperfect results and therefore will require additional resources for verification. Therefore using an algorithm as one of the streams that suggests potentially high-risk offenders for a venue like MATAC might be the way to go. 

Use train_compas_ensemble.py to train a model and then you can use the compas_visualise_models notebook to look at the performance. 


