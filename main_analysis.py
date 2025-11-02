
# main_analysis.py
"""MovieLens Data Analysis - Complete Pure Python Implementation"""
import os, sys, logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.analyzer import MovieAnalyzer
from src.visualizer import InsightsVisualizer
from src.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('analysis.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class MovieLensAnalysisPipeline:
    def __init__(self, data_source: str = "grouplens", dataset: str = "ml-25m"):
        self.data_source, self.dataset = data_source, dataset
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "outputs"
        for d in [self.data_dir / "raw", self.data_dir / "processed",
                  self.output_dir / "plots", self.output_dir / "reports", self.output_dir / "exports"]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized pipeline with source={data_source}, dataset={dataset}")
    def run_full_analysis(self):
        start_time = datetime.now()
        logger.info("="*80); logger.info("Starting MovieLens Data Analysis Pipeline"); logger.info("="*80)
        try:
            logger.info("\n[STAGE 1] Loading Data...")
            data_loader = DataLoader(self.data_dir, self.data_source, self.dataset)
            movies_df, ratings_df = data_loader.load_or_download()
            logger.info(f"✓ Loaded {len(movies_df):,} movies and {len(ratings_df):,} ratings")
            logger.info("\n[STAGE 2] Processing and Cleaning Data...")
            processor = DataProcessor()
            movies_df = processor.clean_movies(movies_df)
            ratings_df = processor.clean_ratings(ratings_df)
            processor.save_parquet(movies_df, ratings_df, self.data_dir / "processed")
            logger.info("✓ Data cleaned and saved to Parquet")
            logger.info("\n[STAGE 3] Performing Statistical Analysis...")
            analyzer = MovieAnalyzer(movies_df, ratings_df)
            top_movies = analyzer.get_top_movies(limit=20, min_ratings=100)
            genre_stats = analyzer.analyze_genre_trends()
            time_series = analyzer.generate_time_series_analysis()
            rating_dist = analyzer.get_rating_distribution()
            user_stats = analyzer.get_user_behavior_stats()
            logger.info("✓ Analysis complete")
            logger.info("\n[STAGE 4] Generating Visualizations...")
            visualizer = InsightsVisualizer(self.output_dir / "plots")
            plots = {
                'rating_distribution': visualizer.plot_rating_distribution(rating_dist),
                'top_movies': visualizer.plot_top_movies(top_movies),
                'genre_popularity': visualizer.plot_genre_popularity(genre_stats),
                'time_series': visualizer.plot_time_series(time_series),
                'user_activity': visualizer.plot_user_activity(user_stats),
                'rating_heatmap': visualizer.plot_rating_heatmap(ratings_df),
                'genre_trends': visualizer.plot_genre_trends_over_time(genre_stats),
                'correlation_matrix': visualizer.plot_correlation_analysis(analyzer)
            }
            logger.info(f"✓ Generated {len(plots)} visualizations")
            logger.info("\n[STAGE 5] Generating Comprehensive Report...")
            report_gen = ReportGenerator(self.output_dir / "reports")
            analysis_results = {
                'top_movies': top_movies, 'genre_stats': genre_stats,
                'time_series': time_series, 'rating_dist': rating_dist,
                'user_stats': user_stats, 'plots': plots,
                'metadata': {
                    'dataset': self.dataset, 'source': self.data_source,
                    'n_movies': len(movies_df), 'n_ratings': len(ratings_df),
                    'n_users': ratings_df['userId'].nunique(),
                    'date_range': (ratings_df['timestamp'].min().strftime('%Y-%m-%d'),
                                   ratings_df['timestamp'].max().strftime('%Y-%m-%d')),
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            report_path = report_gen.generate_html_report(analysis_results)
            logger.info(f"✓ HTML report saved: {report_path}")
            pdf_path = report_gen.generate_pdf_summary(analysis_results)
            logger.info(f"✓ PDF summary saved: {pdf_path}")
            logger.info("\n[STAGE 6] Exporting Results...")
            import pandas as pd
            export_dir = self.output_dir / "exports"
            pd.DataFrame(top_movies).to_csv(export_dir / "top_movies.csv", index=False)
            pd.DataFrame(genre_stats['overall']).to_csv(export_dir / "genre_statistics.csv", index=False)
            pd.DataFrame(time_series['monthly']).to_csv(export_dir / "monthly_trends.csv", index=False)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("\n" + "="*80); logger.info("ANALYSIS COMPLETE!"); logger.info("="*80)
            logger.info(f"Total execution time: {duration:.2f} seconds")
            logger.info(f"\nOutputs saved to: {self.output_dir}")
            logger.info(f"  - Visualizations: {self.output_dir / 'plots'}")
            logger.info(f"  - Reports: {self.output_dir / 'reports'}")
            logger.info(f"  - Exports: {self.output_dir / 'exports'}")
            logger.info("\nOpen the HTML report to view all insights:")
            logger.info(f"  file://{report_path.absolute()}")
            return analysis_results
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True); raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MovieLens Data Analysis Pipeline")
    parser.add_argument('--source', choices=['grouplens', 'huggingface'], default='grouplens')
    parser.add_argument('--dataset', default='ml-25m')
    args = parser.parse_args()
    pipeline = MovieLensAnalysisPipeline(args.source, args.dataset)
    pipeline.run_full_analysis()

if __name__ == "__main__":
    main()
