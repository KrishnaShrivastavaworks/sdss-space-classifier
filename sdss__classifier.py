import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score


def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "star_classification.csv")

    if not os.path.exists(csv_path):
        print("cant find star_classification.csv, make sure its in the same folder")
        exit()

    df = pd.read_csv(csv_path)
    needed = ["u", "g", "r", "i", "z", "redshift", "class"]
    df = df[needed].copy()
    df = df.rename(columns={"z": "z_band"})
    df = df[(df[["u","g","r","i","z_band"]] > -100).all(axis=1)]
    df = df.dropna()
    return df


def add_colours(df):
    df = df.copy()
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z_band"]
    df["u_r"] = df["u"] - df["r"]
    df["g_i"] = df["g"] - df["i"]
    return df


def preprocess(df):
    df = add_colours(df)
    features = ["u", "g", "r", "i", "z_band", "redshift", "u_g", "g_r", "r_i", "i_z", "u_r", "g_i"]
    X = df[features].copy()
    y = df["class"].copy()
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le, features


def plot_results(df, models, X_test, y_test, results, le, features):
    clr = {"STAR": "#4FC3F7", "GALAXY": "#EF9A9A", "QSO": "#A5D6A7"}
    fig = plt.figure(figsize=(20, 22), facecolor="#0d1117")
    fig.suptitle("SDSS Star / Galaxy / Quasar Classifier", fontsize=20, color="white", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    ax0 = fig.add_subplot(gs[0, 0])
    counts = df["class"].value_counts()
    b = ax0.bar(counts.index, counts.values, color=[clr.get(c, "#888") for c in counts.index])
    ax0.bar_label(b, fmt="%d", color="white", fontsize=9)
    style(ax0, "Class Distribution")

    ax1 = fig.add_subplot(gs[0, 1])
    df2 = add_colours(df)
    for cls, grp in df2.groupby("class"):
        ax1.scatter(grp["g_r"], grp["u_g"], s=1, alpha=0.3, label=cls, color=clr.get(cls, "#888"))
    ax1.set_xlabel("g - r")
    ax1.set_ylabel("u - g")
    ax1.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", edgecolor="#444")
    style(ax1, "Colour-Colour Diagram")

    ax2 = fig.add_subplot(gs[0, 2])
    for cls, grp in df.groupby("class"):
        ax2.hist(grp["redshift"].clip(0, 4), bins=60, alpha=0.6, label=cls, color=clr.get(cls, "#888"))
    ax2.set_xlabel("Redshift")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", edgecolor="#444")
    style(ax2, "Redshift Distribution")

    for idx, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, idx])
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        style(ax, "Confusion Matrix - " + name)
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("True", color="white")
        for txt in ax.texts:
            txt.set_color("white")

    ax3 = fig.add_subplot(gs[1, 2])
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    b2 = ax3.bar(names, accs, color=["#4FC3F7", "#A5D6A7"])
    ax3.set_ylim(0.8, 1.0)
    ax3.bar_label(b2, fmt="%.4f", color="white", fontsize=10)
    style(ax3, "Model Accuracy")

    ax4 = fig.add_subplot(gs[2, 0:2])
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=features).sort_values()
    bclr = ["#4FC3F7" if "redshift" in n else "#EF9A9A" for n in imp.index]
    imp.plot(kind="barh", ax=ax4, color=bclr)
    ax4.set_xlabel("Importance")
    style(ax4, "Feature Importances")

    ax5 = fig.add_subplot(gs[2, 2])
    for cls, grp in df.groupby("class"):
        ax5.scatter(grp["redshift"].clip(0, 4), grp["r"], s=1, alpha=0.2, label=cls, color=clr.get(cls, "#888"))
    ax5.set_xlabel("Redshift")
    ax5.set_ylabel("r magnitude")
    ax5.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", edgecolor="#444")
    style(ax5, "r-magnitude vs Redshift")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdss_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("chart saved: " + out)


def predict_mode(model, le, features):
    print("\nenter values to classify a space object")
    print("type exit to quit\n")

    while True:
        try:
            print("enter brightness values:")
            u = input("  u: ").strip()
            if u.lower() == "exit": break
            u = float(u)

            g = input("  g: ").strip()
            if g.lower() == "exit": break
            g = float(g)

            r = input("  r: ").strip()
            if r.lower() == "exit": break
            r = float(r)

            i = input("  i: ").strip()
            if i.lower() == "exit": break
            i = float(i)

            z = input("  z: ").strip()
            if z.lower() == "exit": break
            z = float(z)

            redshift = input("  redshift: ").strip()
            if redshift.lower() == "exit": break
            redshift = float(redshift)

            u_g = u - g
            g_r = g - r
            r_i = r - i
            i_z = i - z
            u_r = u - r
            g_i = g - i

            row = pd.DataFrame([[u, g, r, i, z, redshift, u_g, g_r, r_i, i_z, u_r, g_i]], columns=features)

            pred = model.predict(row)[0]
            proba = model.predict_proba(row)[0]
            label = le.inverse_transform([pred])[0]

            print("\nresult: " + label)
            for cls, prob in zip(le.classes_, proba):
                bar = "#" * int(prob * 30)
                print("  " + cls + ": " + str(round(prob * 100, 1)) + "% " + bar)
            print("")

        except ValueError:
            print("enter a valid number\n")
        except KeyboardInterrupt:
            break


def main():
    df = load_data()
    print("loaded " + str(len(df)) + " rows")
    print(df["class"].value_counts().to_string())

    X, y, le, features = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    }

    print("\ntraining...")
    for name, clf in models.items():
        clf.fit(X_train, y_train)

    results = {}
    for name, clf in models.items():
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"accuracy": acc, "y_pred": y_pred}
        print("\n" + name + " accuracy: " + str(round(acc, 4)))
        print(classification_report(y_test, y_pred, target_names=le.classes_))

    cv = cross_val_score(models["Random Forest"], X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print("cross val: " + str(round(cv.mean(), 4)) + " +/- " + str(round(cv.std(), 4)))

    plot_results(df, models, X_test, y_test, results, le, features)
    predict_mode(models["Random Forest"], le, features)


main()
