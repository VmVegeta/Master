library(usethis)
library("devtools")
#devtools::install_git("https://git.nilu.no/rextapi/frostr.git")
args <- commandArgs(trailingOnly = TRUE)
lat <- args[[1]]
long <- args[[2]]
id <- "0f93e056-5afb-4742-95e3-0f7a3baaed20"
frostr <- frostr::api(httpauth=1,userpwd=paste0(id,":"))
concat <- paste0('nearest(POINT(',long,' ',lat,'))')
station <- frostr$getSources(geometry=concat, format='jsonld')
cat(station[["data"]][[1]][["id"]])
cat(',')
cat(station[["data"]][[1]][["geometry"]][["coordinates"]][[2]])
cat(',')
cat(station[["data"]][[1]][["geometry"]][["coordinates"]][[1]])
cat(',')
cat(station[["data"]][[1]][["distance"]])