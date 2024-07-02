import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class Preprocessor:
    """
    Class to preprocess input data.
    """

    def __init__(self):
        """
        Initialize Preprocessor object with required transformers.
        """
        self.min_max = {}
        self.bins = {}
        self.label_encoder = {}
        self.feature_scaler = MinMaxScaler()
        
    def fit(self, data):
        """
        Fit preprocessing transformers to the source input data.

        Args:
            data (DataFrame): Input data to fit transformers on.
        """
        data = data.copy()

        # Encode the target
        data = self.encode(data)

        # Fit the transformations
        for feature in data.drop(columns=['gender', 'class']).columns:
            # Get the range of continuous features
            self.min_max[feature] = (data[feature].min(), data[feature].max())

            # Determine the bin boundaries
            self.bins[feature] = pd.cut(data[feature], bins=7).unique().categories
            # Assign bin labels
            data[feature+'_bin'] = pd.cut(data[feature], bins=self.bins[feature])
            
            # Label encoding for the binned categories
            self.label_encoder[feature] = LabelEncoder()
            self.label_encoder[feature].fit(data[feature+'_bin'])

            label_encoder = self.label_encoder[feature].transform(data[feature+'_bin'])
            data[feature+'_bin_encoded'] = label_encoder
            
            # Drop the original binned columns
            data.drop([feature+'_bin'], axis=1, inplace=True)
        
        # Fit the scalers
        features, target = self._get_features_and_target(data)
        self.feature_scaler.fit(features)
    
    def transform(self, data, drop_target=True, scale=False):
        """
        Transform input data using the fitted transformers.

        Args:
            data (DataFrame): Input data to transform.
            drop_target (bool): Whether to drop the target variable 'class'.
            scale (bool): Whether to scale the features.

        Returns:
            DataFrame: Transformed data.
        """
        if not drop_target:
            # Encode the target
            data = self.encode(data)

        # Apply the transformations
        for feature in data.drop(columns=['gender', 'class']).columns:
            # Assign bin labels
            data[feature+'_bin'] = pd.cut(data[feature], bins=self.bins[feature])
            # Label encoding for the binned categories
            data[feature+'_bin_encoded'] = self.label_encoder[feature].transform(data[feature+'_bin'])

            # Drop the original binned columns
            data.drop([feature+'_bin'], axis=1, inplace=True)

        if scale:
            # Scale the features
            data = self.scaler(data)

        if drop_target:
            data.drop(columns=['class'], inplace=True)
        
        return data
    
    def fit_transform(self, data):
        """
        Fit transformers to the input data and transform it.

        Args:
            data (DataFrame): Input data to fit and transform.

        Returns:
            DataFrame: Transformed data.
        """
        self.fit(data)
        return self.transform(data)

    def encode(self, data):
        """
        Encode the target ('positive', 'negative' -> 1, 0).

        Args:
            data (DataFrame): Input data.

        Returns:
            DataFrame: Encoded data.
        """
        data = data.copy()
        encoded_class = pd.get_dummies(data['class'], dtype=int, drop_first=True)

        data['class'] = encoded_class

        return data

    def scaler(self, data):
        """
        Scale the features using MinMaxScaler.

        Args:
            data (DataFrame): Input data to scale.

        Returns:
            DataFrame: Scaled data.
        """
        data = data.copy()

        features, target = self._get_features_and_target(data)
        
        scaled_features = self.feature_scaler.transform(features)

        data.loc[:, features.columns] = scaled_features
        
        return data
    
    def inverse_transform_features(self, scaled_features):
        """
        Inverse transform the scaled features to their original scale.

        Args:
            scaled_features (ndarray): Scaled feature values.

        Returns:
            ndarray: Inverse transformed feature values.
        """
        return self.feature_scaler.inverse_transform(scaled_features)
    
    def _get_features_and_target(self, data):
        """
        Helper method to seperate features and target.

        Args:
            data (DataFrame): Input data.

        Returns:
            DataFrame: Features, DataFrame: Target.
        """
        return data.drop(columns=['class']).astype(float), data['class'].astype(object)


# Load environment variables
load_dotenv()
# Get the path to the data from the environment
path_to_data = os.getenv('PATH_TO_DATASET')

# Load data
csv = pd.read_csv(path_to_data)
dataframe = pd.DataFrame(csv)

# Remove the noises
index = dataframe.loc[dataframe['impluse'] > 500].index
dataframe.drop(index=index, inplace=True)

# Initialize the preprocessor
preprocessor = Preprocessor()

# Fit the preprocessor
preprocessor.fit(dataframe)