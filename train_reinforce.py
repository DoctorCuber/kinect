"""Run reinforcement fine-tuning on the captured room pictures."""

if __name__ == "__main__":
    from setup_runtime import ensure

    ensure()

from train_person_model import reinforce_train


def main() -> int:
    return reinforce_train()


if __name__ == "__main__":
    raise SystemExit(main())
