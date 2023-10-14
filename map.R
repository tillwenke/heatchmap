library(ggplot2)
library(ggthemes)
library(viridis) # devtools::install_github("sjmgarnier/viridis)
library(ggmap)
library(scales)
library(grid)
library(dplyr)
library(gridExtra)

dat$cut <- cut(dat$score, breaks=seq(0,100,11), labels=sprintf("Score %d-%d",seq(0, 80, 10), seq(10,90,10)))

orlando <- get_map(location="orlando, fl", source="osm", color="bw", crop=FALSE, zoom=7)

gg <- ggmap(orlando)
gg <- gg + stat_density2d(data=dat, aes(x=lon, y=lat, fill=..level.., alpha=..level..),
                          geom="polygon", size=0.01, bins=5)
gg <- gg + scale_fill_viridis()
gg <- gg + scale_alpha(range=c(0.2, 0.4), guide=FALSE)
gg <- gg + coord_map()
gg <- gg + facet_wrap(~cut, ncol=3)
gg <- gg + labs(x=NULL, y=NULL, title="Score Distribution Across All Schools\n")
gg <- gg + theme_map(base_family="Helvetica")
gg <- gg + theme(plot.title=element_text(face="bold", hjust=1))
gg <- gg + theme(panel.margin.x=unit(1, "cm"))
gg <- gg + theme(panel.margin.y=unit(1, "cm"))
gg <- gg + theme(legend.position="right")
gg <- gg + theme(strip.background=element_rect(fill="white", color="white"))
gg <- gg + theme(strip.text=element_text(face="bold", hjust=0))
gg
