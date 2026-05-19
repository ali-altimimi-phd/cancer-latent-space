# pipeline_logger.R
# -------------------------------------------------------------
# Lightweight timing + logging utility for long pipelines
# Logs to console and optional file with timestamped messages
# -------------------------------------------------------------

start_log <- function(logfile = NULL) {
  log_env <- new.env()
  log_env$start_time <- Sys.time()
  log_env$logfile <- logfile
  log_env$entries <- character()
  
  log_message <- function(msg, section = NULL) {
    timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    prefix <- if (!is.null(section)) glue::glue("[{section}] ") else ""
    full_msg <- glue::glue("{timestamp} | {prefix}{msg}")
    
    message(full_msg)
    log_env$entries <- c(log_env$entries, full_msg)
    if (!is.null(log_env$logfile)) {
      cat(full_msg, file = log_env$logfile, sep = "\n", append = TRUE)
    }
  }
  
  timed <- function(label, expr) {
    log_message(glue::glue("⏱️  Starting: {label}"), section = "TIMER")
    start <- Sys.time()
    result <- force(expr)
    elapsed <- round(difftime(Sys.time(), start, units = "secs"), 2)
    log_message(glue::glue("✅ Done: {label} (elapsed: {elapsed} sec)"), section = "TIMER")
    invisible(result)
  }
  
  list(
    log = log_message,
    timed = timed,
    entries = function() log_env$entries
  )
}
