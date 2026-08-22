# GitHub repositories mentioning Fujifilm X-Trans demosaicing

This inventory was collected from GitHub on 2026-07-18. It combines repository
and README searches for `fujifilm`, `x-trans` or `xtrans`, and `demosaic` or
`demosaicing`; the resulting 24 repositories were deduplicated.

The descriptions below are based on each repository's default-branch README at
the time of the search. They summarize the projects' own claims rather than an
independent audit of their implementations.

* [brainandforce/BayeredImages.jl](https://github.com/brainandforce/BayeredImages.jl) -- A Julia package providing CFA-aware image types and transformations for raw sensor data, currently with bilinear demosaicing. Bayer is supported now, while Fujifilm X-Trans support is planned.

* [d4mr/xtkit](https://github.com/d4mr/xtkit) -- An early-stage Rust toolkit built specifically for Fujifilm X-Trans, with a tested RAF parser and a scaffolded WebGPU processing pipeline. Its roadmap includes a fused GPU raw-development kernel and neural demosaicing trained from X-T5 pixel-shift data.

* [danylo-kelvich/neural-demosaic](https://github.com/danylo-kelvich/neural-demosaic) -- PackedXTransNet is a neural demosaicer for Fujifilm's 6 x 6 X-Trans CFA, reportedly outperforming Markesteijn in PSNR. The repository provides training, tiled PyTorch inference, and CoreML export and inference workflows.

* [EaseUS-Data-Recovery-Wizard-Mac-L/Ease-US-Data-Recovery-Wizard-Mac](https://github.com/EaseUS-Data-Recovery-Wizard-Mac-L/Ease-US-Data-Recovery-Wizard-Mac) -- This README advertises a macOS data-recovery application for deleted or inaccessible files. Its only X-Trans connection is a claim that deep scanning recognizes Fujifilm RAF files with format-specific handling; it is not a demosaicing project.

* [Echostorm44/SharpImage](https://github.com/Echostorm44/SharpImage) -- A pure managed-C# port of ImageMagick targeting .NET 10, with broad image-processing and camera-RAW support. Its RAW decoder claims RAF and X-Trans support alongside bilinear, VNG, and AHD demosaicing.

* [fretboarder/rawview](https://github.com/fretboarder/rawview) -- A viewer that exposes pre-demosaiced sensor mosaics, individual photosites, CFA patterns, and raw histograms rather than developing photographs. It recognizes Fujifilm RAF and the 6 x 6 X-Trans pattern through LibRaw but intentionally performs no demosaicing.

* [Gen-416/dngscan](https://github.com/Gen-416/dngscan) -- A small offline tool that develops RAW files into JPEG using scene measurement, LibRaw, and darktable-derived AgX tone mapping. Fujifilm X-Trans files retain LibRaw's native demosaicing path rather than using an algorithm implemented by dngscan.

* [GoldJohnKing/RawAlchemyCpp](https://github.com/GoldJohnKing/RawAlchemyCpp) -- A C++ RAW-processing library and CLI for cinematic log conversion, exposure handling, lens correction, and 3D-LUT grading. It includes darktable-derived Markesteijn and RCD X-Trans demosaicers plus optional neural inference using x-veon ONNX weights, whose unresolved upstream licensing is explicitly noted.

* [imazen/zenraw](https://github.com/imazen/zenraw) -- A safe, pure-Rust camera-RAW and DNG decoder with interchangeable backends and display-ready or scene-linear output. Its `rawler` backend handles Fujifilm RAF and currently provides bilinear X-Trans demosaicing.

* [jgaugustine/cp-DemosaicLab](https://github.com/jgaugustine/cp-DemosaicLab) -- An educational React and TypeScript application for visualizing CFA mosaics, reconstruction errors, and simple demosaicing algorithms. It implements a basic Fujifilm 6 x 6 X-Trans interpolation but requires proprietary RAF files to be converted to DNG first.

* [jgaugustine/DemosaicLab](https://github.com/jgaugustine/DemosaicLab) -- This has the same README and described functionality as `cp-DemosaicLab`: an interactive demosaicing teaching tool with synthetic and real-DNG modes. Its algorithms include nearest-neighbor, bilinear, and a basic TypeScript X-Trans implementation.

* [joaopfsilva/patina](https://github.com/joaopfsilva/patina) -- A client-side RAW photo editor delivered as both a web application and a native macOS application. It uses LibRaw for compressed Fujifilm RAF and X-Trans demosaicing, with an 8-bit WASM path on the web and a 16-bit native pipeline.

* [kyakaze/hassy-x-fuji](https://github.com/kyakaze/hassy-x-fuji) -- A Windows application that converts Hasselblad X2D II RAW files into DNGs identified as Fuji GFX 100S II, unlocking Adobe's Fuji camera profiles. It deliberately chooses a Bayer-based GFX identity because spoofing an X-Trans camera would make Lightroom apply the wrong demosaic pattern.

* [Mauricio-xx/FujiCore-CLI](https://github.com/Mauricio-xx/FujiCore-CLI) -- A RAW processor aimed at Fujifilm X-Trans III cameras, combining film simulations, LUT grading, grain, tone curves, and batch processing. Its C++ engine claims an RCD demosaicer optimized specifically for the X-Trans III 6 x 6 pattern.

* [MAzmi25/Filter_Lengkap](https://github.com/MAzmi25/Filter_Lengkap) -- The README is actually a large Indonesian general-knowledge quiz embedded in HTML rather than documentation for image-processing software. It appeared in the search because its question bank defines demosaicing and identifies X-Trans as a Fujifilm CFA; it implements no RAW processing.

* [naorunaoru/demosaic](https://github.com/naorunaoru/demosaic) -- A pure-Rust, `no_std`-compatible demosaicing library supporting Bayer, Quad Bayer, and X-Trans sensor data. Its X-Trans algorithms include bilinear, one- and three-pass Markesteijn, and DHT, while RAW parsing and color conversion remain out of scope.

* [naorunaoru/x-veon](https://github.com/naorunaoru/x-veon) -- A neural demosaicing project whose CFA-agnostic U-Net handles both Bayer and X-Trans mosaics using sensor values plus color-mask channels. It also includes an offline browser RAW developer powered by ONNX WebGPU and tested mainly with Fujifilm RAF and Sony ARW files.

* [northernpaws/phototools](https://github.com/northernpaws/phototools) -- A C++ floating-point RAW developer derived from dcraw and darktable concepts and tested primarily on Fujifilm X-Trans files. Its X-Trans interpolation is adapted from dcraw and followed by color-space conversion into a displayable image.

* [rymuelle/RawHandler](https://github.com/rymuelle/RawHandler) -- A Python wrapper around rawpy for extracting normalized sensor arrays and preparing RAW data for neural-network training. It currently implements several Bayer demosaicers, while Fujifilm X-Trans support is explicitly listed as work in progress.

* [saswatamcode/raw_mosaic](https://github.com/saswatamcode/raw_mosaic) -- A small LibRaw-based program for visualizing grayscale channels, color mosaics, and repeating CFA tiles from camera RAW files. It demonstrates Fujifilm X-M5 X-Trans IV data but explicitly stops before demosaicing, white balance, or color conversion.

* [Scdouglas1999/NINA-Fujifilm-Native-Plugin](https://github.com/Scdouglas1999/NINA-Fujifilm-Native-Plugin) -- A Windows plugin providing direct USB control and RAW capture from Fujifilm cameras in the N.I.N.A. astronomy application. For X-Trans cameras it generates a synthetic Bayer preview with selectable quality while preserving the original RAF sensor data and relevant FITS metadata.

* [tulensrma/rafinery](https://github.com/tulensrma/rafinery) -- A batch tool that turns folders of RAW photographs into punchy JPEG baselines intended for later hand editing. It targets Fujifilm shooters but delegates X-Trans and Bayer demosaicing, highlight recovery, color, and lens correction to `darktable-cli`.

* [vdavid/prvw](https://github.com/vdavid/prvw) -- A lightweight, GPU-accelerated macOS image viewer written in Rust with support for major camera RAW formats. Its current Fujifilm RAF path uses a relatively soft bilinear X-Trans demosaic through `rawler`.

* [z16166/SimpleImageViewer](https://github.com/z16166/SimpleImageViewer) -- A cross-platform Rust image viewer supporting numerous raster, HDR, Photoshop, and camera-RAW formats. It offers CPU and GPU high-quality RAW preview modes, but Fujifilm X-Trans data currently falls back to CPU processing.
