"""Phase 10.4: isolated win-oriented training, no champion promotion.

Train baseline and experimental variants in separate directories with the
same seed. Preserve the source checkpoint; evaluate on fresh held-out seeds.
"""
import argparse
from pathlib import Path

from .train import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Frozen neural checkpoint to initialize both variants")
    parser.add_argument("--variant", choices=("baseline", "win"), required=True)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--eval-games", type=int, default=80)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--competitive-rounds", type=int, default=16)
    parser.add_argument("--competitive-patience", type=int, default=2)
    parser.add_argument("--win-weight", type=float, default=2.0)
    parser.add_argument("--relative-weight", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not args.candidate.is_file():
        parser.error(f"checkpoint does not exist: {args.candidate}")
    if args.episodes < 1 or args.eval_games < 1 or args.eval_interval < 1:
        parser.error("episodes, eval-games and eval-interval must be positive")
    if args.learning_rate <= 0 or args.win_weight < 0 or args.relative_weight < 0 or args.anchor_weight < 0 or args.competitive_rounds < 2 or args.competitive_patience < 1:
        parser.error("learning rate must be positive and weights nonnegative")
    out = args.out or Path("runs/phase104") / args.variant
    if out.resolve() == args.candidate.parent.resolve():
        parser.error("output must not overwrite the source checkpoint directory")
    train(
        episodes=args.episodes, seed=args.seed, out=str(out),
        eval_games=args.eval_games, eval_interval=args.eval_interval,
        learning_rate=args.learning_rate, init_checkpoint=args.candidate,
        finetune_anchor_weight=args.anchor_weight,
        competitive_rounds=args.competitive_rounds,
        competitive_patience=args.competitive_patience,
        terminal_win_weight=1.0 if args.variant == "baseline" else args.win_weight,
        relative_score_weight=0.0 if args.variant == "baseline" else args.relative_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
