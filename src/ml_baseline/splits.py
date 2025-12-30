from sklearn.model_selection import train_test_split


def split_data(df, cfg):
    return train_test_split(
        df,
        train_size=cfg.train_size,
        random_state=cfg.session_id,
        stratify=df[cfg.target],
    )
