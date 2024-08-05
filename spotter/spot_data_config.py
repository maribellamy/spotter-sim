from datetime import date


class SpotConfig:
    def __init__(self, flavor: str, tag: str, points: list) -> None:
        assert isinstance(flavor, str)
        assert isinstance(tag, str)
        assert isinstance(points, list)
        self.flavor: str = flavor
        self.tag: str = tag
        self.points: list = points

    def __str__(self) -> str:
        return f"SpotConfig(flavor={self.flavor}, tag={self.tag}, points={self.points})"

    def __repr__(self) -> str:
        return self.__str__()


class SpotData:
    def __init__(self, d: date, step: str) -> None:
        assert isinstance(d, date)
        assert isinstance(step, str)
        self.date: date = d
        self.step: str = step
        self.spot_configs: dict[str, SpotConfig] = dict()
        self.data: dict[str, str | None] = dict()

    def __str__(self) -> str:
        return (
            f"SpotData(date={self.date}, step={self.step}, configs={self.spot_configs})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def add_spot_config(self, *, spot_config: SpotConfig) -> None:
        assert isinstance(spot_config, SpotConfig)
        assert self.spot_configs.get(spot_config.flavor, None) is None
        self.spot_configs[spot_config.flavor] = spot_config

    def add_data(self, *, flavor: str, data: str | None) -> None:
        assert isinstance(flavor, str)
        assert data is None or isinstance(data, str)
        assert self.spot_configs.get(flavor, None) is not None
        assert self.data.get(flavor, None) is None
        self.data[flavor] = data

    def get_tag(self, *, flavor: str) -> str:
        assert isinstance(flavor, str)
        assert self.spot_configs.get(flavor, None) is not None
        return self.spot_configs[flavor].tag

    def get_points(self, *, flavor: str) -> list[str]:
        assert isinstance(flavor, str)
        assert self.spot_configs.get(flavor, None) is not None
        return self.spot_configs[flavor].points

    def get_data(self, *, flavor: str) -> str:
        assert isinstance(flavor, str)
        assert self.data.get(flavor, None) is not None
        return self.data[flavor]
