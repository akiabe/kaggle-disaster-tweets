import config
import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)

    # create a new column kfold and fill it with -1
    df["kfold"] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch targets
    y = df.target.values

    # instantiate the kfold from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)