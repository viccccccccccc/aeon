def save_run_info_txt(combined_output_file, all_results):
    
    with open(combined_output_file, 'w') as f:
        f.write(f"Benchmark: {all_results['benchmark']}\n\n")
        for idx, result in enumerate(all_results):
            f.write(f"--- Ergebnisse für Kombination ({idx + 1}): Spikelet={result['use_spikelet']}, Downsampling={result['use_downsampling']} ---\n")
            f.write(f"Score: {result['score']}\n")
            f.write(f"Time for ClaSP: {result['clasp_fit_predict_time']:.4f} seconds\n")
            f.write(f"Time for Spikelet: {result['transformation_time']:.4f} seconds\n")
            f.write(f"Time for Downsampling: {result['downsampling_time']:.4f} seconds\n")
            f.write(f"Window Size: {result['window_size']}\n")
            f.write(f"Time Series length: {result['time_series_length']}\n")
            f.write(f"Original Time Series Length: {result['original_time_series_length']}\n")
            f.write(f"Peak memory usage ClaSP: {result['clasp_memory_peak'] / 1024:.2f} KB\n")
            f.write(f"Current memory usage ClaSP: {result['clasp_memory_current'] / 1024:.2f} KB\n")
            f.write(f"Peak memory usage Spikelet: {result['spikelet_memory_peak'] / 1024:.2f} KB\n")
            f.write(f"Current memory usage Spikelet: {result['spikelet_memory_current'] / 1024:.2f} KB\n")
            f.write(f"Peak memory usage Downsampling: {result['downsampling_memory_peak'] / 1024:.2f} KB\n")
            f.write(f"Current memory usage Downsampling: {result['downsampling_memory_peak'] / 1024:.2f} KB\n\n")