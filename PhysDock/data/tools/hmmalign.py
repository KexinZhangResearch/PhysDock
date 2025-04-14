import os
import subprocess
from typing import Optional, Sequence
import logging

from . import parsers
from . import hmmbuild
from . import utils


class Hmmalign(object):
    def __init__(
            self,
            *,
            hmmbuild_binary_path: str,
            hmmalign_binary_path: str,
    ):
        self.binary_path = hmmalign_binary_path
        self.hmmbuild_runner = hmmbuild.Hmmbuild(binary_path=hmmbuild_binary_path)

    @property
    def output_format(self) -> str:
        return 'sto'

    @property
    def input_format(self) -> str:
        return 'sto'

    def realign_sto_with_fasta(self, input_fasta_path, input_sto_path, output_sto_path: Optional = None) -> str:
        delete_out = False if output_sto_path is not None else True
        with utils.tmpdir_manager() as query_tmp_dir:
            hmm_output_path = os.path.join(query_tmp_dir, 'query.hmm')
            output_sto_path = os.path.join(query_tmp_dir,
                                           "realigned.sto") if output_sto_path is None else output_sto_path
            with open(input_fasta_path, "r") as f:
                hmm = self.hmmbuild_runner.build_rna_profile_from_fasta(f.read())
            with open(hmm_output_path, 'w') as f:
                f.write(hmm)

            cmd = [
                self.binary_path,
                '--rna',  # Don't include the alignment in stdout.
                '--mapali', input_fasta_path,
                "-o", output_sto_path,
                hmm_output_path,
                input_sto_path
            ]
            # print(cmd)
            # print(" ".join(cmd))
            logging.info('Launching sub-process %s', cmd)
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with utils.timing(
                    f'hmmsearch  query'):
                stdout, stderr = process.communicate()
                retcode = process.wait()

            if retcode:
                raise RuntimeError(
                    'hmmsearch failed:\nstdout:\n%s\n\nstderr:\n%s\n' % (
                        stdout.decode('utf-8'), stderr.decode('utf-8')))
            if delete_out:
                with open(output_sto_path) as f:
                    out_msa = f.read()
        if delete_out:
            return out_msa
