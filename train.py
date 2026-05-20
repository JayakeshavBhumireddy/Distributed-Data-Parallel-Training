from ddp_vision.config import parse_args


def main() -> None:
    config = parse_args()
    from ddp_vision.trainer import run_training

    run_training(config)


if __name__ == "__main__":
    main()
