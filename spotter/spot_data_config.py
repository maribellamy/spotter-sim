from datetime import date


class spot_config:
    def __init__(self,
                 flavor: str,
                 tag: str,
                 components: list[str]) -> None:
        assert isinstance(flavor, str)
        assert isinstance(tag, str)
        assert isinstance(components, list)
        assert len(components) != 0 and isinstance(components[0], str)
        self.flavor: str = flavor
        self.tag: str = tag
        self.components: list[str] = components

    def __str__(self) -> str:
        return f"SpotConfig(flavor={self.flavor}, tag={self.tag}, components={self.components})"

    def __repr__(self) -> str:
        return self.__str__()


class spot_data:
    def __init__(self,
                 d: date,
                 step: str) -> None:
        assert isinstance(d, date)
        assert isinstance(step, str)
        self.date: date = d
        self.step: str = step
        self.spot_configs: dict[str, spot_config] = dict()
        self.data: dict[str, str | None] = dict()

    def __str__(self) -> str:
        return f"SpotData(date={self.date}, step={self.step}, configs={self.spot_configs})"

    def __repr__(self) -> str:
        return self.__str__()

    def addSpotConfiguration(self,
                             *,
                             spotConfig: spot_config) -> None:
        assert isinstance(spotConfig, spot_config)
        assert self.spot_configs.get(spotConfig.flavor, None) is None
        self.spot_configs[spotConfig.flavor] = spotConfig

    def addData(self,
                *,
                flavor: str,
                data: str | None) -> None:
        assert isinstance(flavor, str)
        assert data is None or isinstance(data, str)
        assert self.spot_configs.get(flavor, None) is not None
        assert self.data.get(flavor, None) is None
        self.data[flavor] = data

    def getTag(self,
               *,
               flavor: str) -> str:
        assert isinstance(flavor, str)
        assert self.spot_configs.get(flavor, None) is not None
        return self.spot_configs[flavor].tag

    def getComponents(self,
                      *,
                      flavor: str) -> list[str]:
        assert isinstance(flavor, str)
        assert self.spot_configs.get(flavor, None) is not None
        return self.spot_configs[flavor].components

    def getData(self,
                *,
                flavor: str) -> str:
        assert isinstance(flavor, str)
        assert self.data.get(flavor, None) is not None
        return self.data[flavor]
