#' Download CEL Files from FTP Server
#'
#' Downloads Affymetrix CEL files from an FTP server using RCurl. Supports dry-run mode
#' and limiting downloads for testing. Designed to be integrated into a pipeline logger.
#'
#' @param ftp_base The base URL of the FTP server.
#' @param local_cel_dir Directory where CEL files will be saved.
#' @param download_limit Optional integer to limit the number of files downloaded. Default: 3.
#' @param dry_run Logical. If TRUE, simulates downloads. Default: TRUE.
#' @param logger Optional logging object (e.g., from `pipeline_logger.R`). If NULL, uses `message()`.
#'
#' @return Invisibly returns list of downloaded files (or would-be downloads if dry-run).
#' @export
download_cel_files <- function(ftp_base,
                               local_cel_dir,
                               download_limit = 3,
                               dry_run = TRUE,
                               logger = NULL) {
  logf <- function(msg) {
    if (!is.null(logger)) logger$log(msg) else message(msg)
  }
  
  logf("🚀 Starting FTP CEL file download")
  logf("📥 Listing files on FTP server...")
  
  ftp_listing <- tryCatch({
    getURL(ftp_base, ftp.use.epsv = FALSE, dirlistonly = TRUE)
  }, error = function(e) {
    logf(glue::glue("❌ FTP listing failed: {e$message}"))
    stop("FTP listing failed.")
  })
  
  cel_files <- unlist(strsplit(ftp_listing, "\r*\n"))
  cel_files <- cel_files[grepl("\\.CEL$", cel_files)]
  
  if (!length(cel_files)) {
    logf("⚠️ No CEL files found on FTP.")
    return(invisible(NULL))
  }
  
  # Limit downloads for testing
  if (!is.null(download_limit)) {
    cel_files <- head(cel_files, download_limit)
    logf(glue::glue("🧪 Test mode: limiting to {length(cel_files)} CEL files."))
  } else {
    logf(glue::glue("📄 Found {length(cel_files)} CEL files to download."))
  }
  
  downloaded <- character()
  for (file in cel_files) {
    destfile <- file.path(local_cel_dir, file)
    
    if (dry_run) {
      logf(glue::glue("🧪 Would download: {file} → {destfile}"))
      next
    }
    
    if (file.exists(destfile)) {
      logf(glue::glue("✅ Skipping {file} (already exists)"))
      next
    }
    
    logf(glue::glue("⬇️ Downloading {file}..."))
    tryCatch({
      download.file(glue::glue("{ftp_base}{file}"), destfile = destfile, mode = "wb")
      logf(glue::glue("✅ Downloaded: {file}"))
      downloaded <- c(downloaded, file)
    }, error = function(e) {
      logf(glue::glue("❌ Failed to download {file}: {e$message}"))
    })
  }
  
  invisible(downloaded)
}
