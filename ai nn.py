
from pathlib import Path
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class IrisClassifier(nn.Module):
    def __init__(self, in_features=4, h1=16, h2=16,
                 h3=34, out_features=2):
        super().__init__()

        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


def similarity(a, b):
    a = str(a).strip().lower()
    b = str(b).strip().lower()

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def get_features(player1, player2, columns):
    return [
        similarity(player1[col], player2[col])
        for col in columns
    ]


def main() -> None:
    torch.manual_seed(41)

    folder = Path(__file__).parent
    data_path = folder / "players.csv"
    labels_path = folder / "alt_labels.csv"

    if not data_path.exists():
        print("ERROR: players.csv not found.")
        return

    if not labels_path.exists():
        print("ERROR: alt_labels.csv not found.")
        print("Required columns: player1,player2,is_alt")
        return

    data = pd.read_csv(data_path, dtype=str).fillna("")
    data.columns = data.columns.str.strip().str.lower()

    labels = pd.read_csv(labels_path, dtype=str)
    labels.columns = labels.columns.str.strip().str.lower()

    if "player id" not in data.columns:
        print("ERROR: Missing 'player id' column.")
        return

    if not {"player1", "player2", "is_alt"}.issubset(labels.columns):
        print("ERROR: Incorrect alt_labels.csv columns.")
        return

    if data["player id"].duplicated().any():
        print("ERROR: Duplicate player IDs in rblx.csv.")
        return

    # Use behavioral fields, not player IDs or IP addresses.
    feature_prefixes = ["mouse", "keyb", "chat", "time"]
    columns = []

    for prefix in feature_prefixes:
        match = next(
            (c for c in data.columns if c.startswith(prefix)),
            None
        )
        if match is not None:
            columns.append(match)

    if not columns:
        print("ERROR: No activity columns found.")
        return

    print("Using columns:", columns)

    players = data.set_index("player id")

    features = []
    targets = []
    pairs = []

    for _, row in labels.iterrows():
        p1 = str(row["player1"]).strip()
        p2 = str(row["player2"]).strip()

        if p1 not in players.index or p2 not in players.index:
            continue

        if p1 == p2:
            continue

        if row["is_alt"] not in ("0", "1"):
            print("ERROR: is_alt must contain only 0 or 1.")
            return

        features.append(
            get_features(players.loc[p1], players.loc[p2], columns)
        )
        targets.append(int(row["is_alt"]))
        pairs.append((p1, p2))

    if len(features) < 10 or len(set(targets)) < 2:
        print("ERROR: Need more confirmed examples of both classes.")
        return

    # Remove duplicate account pairs, including reversed pairs.
    unique = {}
    for pair, feature, target in zip(pairs, features, targets):
        key = tuple(sorted(pair))
        if key in unique and unique[key][1] != target:
            print("ERROR: Conflicting labels for", key)
            return
        unique[key] = (feature, target)

    features = [item[0] for item in unique.values()]
    targets = [item[1] for item in unique.values()]

    counts = pd.Series(targets).value_counts()

    if counts.min() < 2 or len(features) < 10:
        print("ERROR: Need at least 10 unique pairs")
        print("and at least 2 examples of each class.")
        return

    # Split confirmed examples into training and test sets.
    # For a real evaluation, split by account groups to avoid
    # the same accounts leaking into both sets.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42,
        stratify=targets
    )

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    classifier = IrisClassifier(
        in_features=x_train.shape[1],
        out_features=2
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=0.001
    )

    print("\nTraining PyTorch AI...\n")

    for epoch in range(900):
        classifier.train()

        predictions = classifier(x_train)
        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 899:
            classifier.eval()

            with torch.no_grad():
                train_predictions = classifier(x_train).argmax(dim=1)
                train_accuracy = (
                    (train_predictions == y_train)
                    .float().mean().item()
                )

                test_predictions = classifier(x_test).argmax(dim=1)
                test_accuracy = (
                    (test_predictions == y_test)
                    .float().mean().item()
                )

            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Accuracy: {train_accuracy:.1%} | "
                f"Test Accuracy: {test_accuracy:.1%}"
            )

    classifier.eval()

    with torch.no_grad():
        final_train = classifier(x_train).argmax(dim=1)
        final_test = classifier(x_test).argmax(dim=1)

        train_accuracy = (
            (final_train == y_train).float().mean().item()
        )
        test_accuracy = (
            (final_test == y_test).float().mean().item()
        )

    print("\n========== FINAL RESULTS ==========")
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy:     {test_accuracy:.2%}")
    print("===================================\n")

    print("Checking possible alt accounts...\n")

    results = []

    with torch.no_grad():
        for p1, p2 in combinations(players.index, 2):
            pair_features = get_features(
                players.loc[p1],
                players.loc[p2],
                columns
            )

            input_tensor = torch.tensor(
                [pair_features],
                dtype=torch.float32
            )

            output = classifier(input_tensor)

            score = F.softmax(output, dim=1)[0, 1].item()

            results.append({
                "player1": p1,
                "player2": p2,
                "model_score": round(score, 4)
            })

    results.sort(
        key=lambda result: result["model_score"],
        reverse=True
    )

    for result in results:
        print(
            f"Player {result['player1']} "
            f"<-> Player {result['player2']} "
            f"| Model score: {result['model_score']:.1%}"
        )

    output_path = folder / "possible_alts.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
