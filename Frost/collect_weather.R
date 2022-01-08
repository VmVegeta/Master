library(usethis)
library("devtools")
#devtools::install_git("https://git.nilu.no/rextapi/frostr.git")
args <- commandArgs(trailingOnly = TRUE)
station_id <- args[[1]]
start_date <- args[[2]]
end_date <- args[[3]]

id <- "0f93e056-5afb-4742-95e3-0f7a3baaed20"
frostr <- frostr::api(httpauth=1,userpwd=paste0(id,":"))
time <- paste0(start_date, '/', end_date)
res <- frostr$observations(sources=station_id,
                        referencetime=time,
                        elements='wind_speed',
                        format='jsonld')

for(observation in res[["data"]]){
    cat(observation[["referenceTime"]])
    cat(',')
    cat(observation[["observations"]][[1]][["value"]])
    cat(';')
}
