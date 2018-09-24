import click
from experiment import Experiment

@click.group()
def cli():
	pass


@cli.command('train', help='Train model')
@click.option('--db', default='10', help=u'Database that will be used: Cifar10 (10), Cifar100 (100) or EMNIST.')
@click.option('--net_type', '-n', default='vgg19',
			  help=u'Net model that will be used. Must be one of: vgg19, resnet56, resnet110')
@click.option('--batch_size', '-b', default=128, help=u'Batch size')
@click.option('--epochs', '-e', default=100, help=u'Number of epochs')
@click.option('--checkpoint_dir', '-d', required=True, help=u'Checkpoint files directory')
@click.option('--activation', '-a', default='relu', help=u'Activation function')
@click.option('--spp_alpha', default=0.2, help=u'Alpha value for spp transfer function')
@click.option('--lr', default=0.1, help=u'Learning rate')
@click.option('--momentum', '-m', default=0.1, help=u'Momentum for optimizer')
@click.option('--rep', '-r', default=1, help=u'Repetitions for this execution.')
@click.option('--dropout', '-o', default=0.0, help=u'Drop rate.')
def train(db, net_type, batch_size, epochs, checkpoint_dir, activation, spp_alpha, lr, momentum, rep, dropout):
	for execution in range(1, rep + 1):
		experiment = Experiment('', db, net_type, batch_size, epochs, checkpoint_dir, activation, spp_alpha, lr, momentum, dropout)
		experiment.set_auto_name()
		experiment.checkpoint_dir = "{}/{}/{}".format(checkpoint_dir, experiment.get_auto_name(), execution)
		experiment.run()


if __name__ == '__main__':
	cli()