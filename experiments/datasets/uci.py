# Imports
import pandas as pd


# Dataset loader function
def load(name, seed=None):
    """
    This function loads the UCI datasets from their respective CSV-files, specified by the `name` input.
    
      - Datasets: boston / concrete / energy / kin8nm / naval / powerplant / wine / yacht
    """
    
    if name == 'boston':
        # Meta-data
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                        'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        y_label = 'MEDV'
        # Load dataset
        loc = './datasets/' + name + '.csv'
        raw_dataset = pd.read_csv(loc, names=column_names, sep=' ', skipinitialspace=True)

    elif name == 'concrete':
        # Meta-data
        column_names = ['Cement', 'Slag', 'Fly Ash', 'Water', 'Superplasticizer', 
                        'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Compressive Strength']
        y_label = 'Compressive Strength'
        # Load dataset
        loc = './datasets/' + name + '.csv'
        raw_dataset = pd.read_csv(loc, names=column_names, sep=',', skipinitialspace=True)

    elif name == 'energy':
        # Meta-data
        column_names = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 
                        'Orientation', 'Glazing Area', 'Glazing Distribution', 'Heating Load', 'Cooling Load']
        y_labels = ['Heating Load', 'Cooling Load']
        # Load dataset
        loc = './datasets/' + name + '.csv'
        raw_dataset = pd.read_csv(loc, names=column_names, sep=',', skipinitialspace=True)

    elif name == 'kin8nm':
        # Meta-data
        column_names = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 
                        'theta7', 'theta8', 'y']
        y_label = 'y'
        # Load dataset
        loc = './datasets/' + name + '.csv'
        raw_dataset = pd.read_csv(loc, names=column_names, sep=',', skipinitialspace=True)

    elif name == 'naval':
        # Meta-data
        column_names = ['lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2',
                        'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf', 'Compressor', 'Turbine']
        y_labels = ['Compressor', 'Turbine']
        # Load dataset
        loc = './datasets/' + name + '.csv'
        raw_dataset = pd.read_csv(loc, names=column_names, sep=',', skipinitialspace=True)

    elif name == 'powerplant':
        # Meta-data
        column_names = ['AT', 'V', 'AP', 'RH', 'PE']
        y_label = 'PE'
        # Load dataset
        loc = './datasets/' + name + '.csv'
        raw_dataset = pd.read_csv(loc, names=column_names, sep=',', skipinitialspace=True)

    elif name == 'wine':
        # Meta-data
        column_names = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 'Chlorides',
                        'Free SO2', 'Total SO2', 'Density', 'pH', 'Sulphates', 'Alcohol', 'Quality']
        y_label = 'Quality'
        # Load dataset
        loc = './datasets/' + name + '.csv'
        raw_dataset = pd.read_csv(loc, names=column_names, sep=',', skipinitialspace=True)
        raw_dataset[y_label] = raw_dataset[y_label].astype(float)

    elif name == 'yacht':
        # Meta-data
        column_names = ['Position', 'Prismatic', 'Displacement', 'Beam-draught', 'Length-beam', 
                        'Froude', 'Resistance']
        y_label = 'Resistance'
        # Load dataset
        loc = './datasets/' + name + '.csv'
        raw_dataset = pd.read_csv(loc, names=column_names, sep=' ', skipinitialspace=True)

    # Copy dataset and drop NaNs
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    
    # Split into test and train
    train_dataset = dataset.sample(frac=0.9, random_state=seed)
    test_dataset  = dataset.drop(train_dataset.index)

    # Create features ...
    x_train = train_dataset.copy()
    x_test  = test_dataset.copy()
    # ... and labels
    if (name == 'energy') or (name == 'naval'):
        y_train = x_train[y_labels].copy()
        x_train = x_train.drop(y_labels, axis=1)
        y_test  = x_test[y_labels].copy()
        x_test  = x_test.drop(y_labels, axis=1)
    else: 
        y_train = x_train.pop(y_label)
        y_test  = x_test.pop(y_label)

    # Return dataset, only single 'x' and 'y'
    return (pd.concat([x_train, x_test]), pd.concat([y_train, y_test]))