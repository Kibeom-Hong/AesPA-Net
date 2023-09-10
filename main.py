from __future__ import print_function
import argparse, os


def str2bool(v):
	if v.lower() in ('true', 't'):
		return True
	elif v.lower() in ('false', 'f'):
		return False


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='AesPA_Net')
	parser.add_argument('--comment', default='')
	parser.add_argument('--output_image_path', default='./results')
	parser.add_argument('--content_dir', type=str, default='../../dataset/MSCoCo', help='Content data path to train the network')
	parser.add_argument('--style_dir', type=str, default='../../dataset/wikiart', help='Content data path to train the network')

	######For train arguments#####
	parser.add_argument('--train_type', type=str, default='split')
	parser.add_argument('--type', type=str, default='train')
	parser.add_argument('--max_iter', type=int, default=1000)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--check_iter', type=int, default=100)
	parser.add_argument('--imsize', type=int, default=512)
	parser.add_argument('--cropsize', type=int, default=256)
	parser.add_argument('--num_workers', type=int, default=12)
	parser.add_argument('--cencrop', action='store_true', default=False)
	parser.add_argument('--train_result_dir', type=str, default='./train_results', help='Content data path to train the network')

	######For test arguments#####
	parser.add_argument('--test_result_dir', type=str, default='./test_results', help='Test results')
	parser.add_argument('--test_content_segment', type=str, default='./test_images/content/')
	parser.add_argument('--test_p_reference_segment', type=str, default='./test_images/p_reference/')
	parser.add_argument('--test_iter', type=int, default=0)

	
	args = parser.parse_args()

	
	
	from baseline import Baseline as Baseline
	model = Baseline(args)

	if args.type == 'train':
		model.train()
	elif args.type == 'test':
		model.test(args)
	elif args.type == 'eval':
		model.eval(args)
	elif args.type == 'CF':
		model.content_fidelity(args)
	elif args.type == 'analysis':
		model.analysis(args)

		


