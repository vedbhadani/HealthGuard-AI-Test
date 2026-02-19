# Milestone 1 - ML Pipeline UML Diagram

```mermaid
classDiagram

class PatientInput {
    +int Pregnancies
    +float Glucose
    +float BloodPressure
    +float SkinThickness
    +float Insulin
    +float BMI
    +float DiabetesPedigreeFunction
    +int Age
}

class DataPreprocessor {
    +handleMissingValues()
    +standardizeFeatures()
}

class LogisticRegressionModel {
    +train()
    +predict()
    +predict_proba()
}

class RiskEvaluator {
    +calculateProbability()
    +categorizeRisk()
}

class ExplainabilityModule {
    +calculateFeatureContribution()
}

class StreamlitUI {
    +collectInput()
    +displayPrediction()
    +displayFeatureContribution()
}

PatientInput --> DataPreprocessor
DataPreprocessor --> LogisticRegressionModel
LogisticRegressionModel --> RiskEvaluator
RiskEvaluator --> ExplainabilityModule
ExplainabilityModule --> StreamlitUI
