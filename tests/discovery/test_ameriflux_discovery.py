import os
import glob

import pytest

import act
from act.discovery.ameriflux import PolicyWarning


# Place your username and token here
user_id = os.getenv('AMERIFLUX_USERNAME')
user_email = os.getenv('AMERIFLUX_EMAIL')

if user_id is not None and user_email is not None:
    if len(user_id) == 0 and len(user_email) == 0:
        ameriflux_available = False

@pytest.mark.skipif(not ameriflux_available, reason="Can't download files")
def test_download_ameriflux_data():
    if not os.path.isdir(os.getcwd() + '/data/'):
        os.makedirs(os.getcwd() + '/data/')

    # Place your username and token here
    user_id = os.getenv('AMERIFLUX_USERNAME')
    user_email = os.getenv('AMERIFLUX_EMAIL')

    if user_id is not None and user_email is not None:
        if len(user_id) == 0 and len(user_email) == 0:
            return
        kwargs = {}
        kwargs['is_test'] = True
        site_ids = ['US-CU1']
        data_product = 'BASE-BADM'
        data_policy = 'CCBY4.0'
        data_variant = 'FULLSET'
        intended_use = 'other'
        description = 'I intend to use this data for research'
        out_dir = os.getcwd() + '/data/'

        # Test if CCBY4.0 data policy is printed
        with pytest.warns(PolicyWarning):
            results = act.discovery.ameriflux.download_ameriflux_data(
                user_id,
                user_email,
                data_product='BASE-BADM',
                data_policy=data_policy,
                data_variant=data_variant,
                site_ids=['US-CU1'],
                agree_policy=True,
                intended_use=intended_use,
                description=description,
                out_dir=out_dir,
                **kwargs,
            )

        # Test if legacy data policy is printed
        with pytest.warns(PolicyWarning):
            results = act.discovery.ameriflux.download_ameriflux_data(
                user_id,
                user_email,
                data_product='BASE-BADM',
                data_policy='LEGACY',
                data_variant=data_variant,
                site_ids=['US-CU1'],
                agree_policy=True,
                intended_use=intended_use,
                description=description,
                out_dir=out_dir,
                **kwargs,
            )

        # Test if BASE-BADM files are downloaded
        results = act.discovery.ameriflux.download_ameriflux_data(
            user_id,
            user_email,
            data_product=data_product,
            data_policy=data_policy,
            data_variant=data_variant,
            site_ids=site_ids,
            agree_policy=True,
            intended_use=intended_use,
            description=description,
            out_dir=out_dir,
            **kwargs,
        )

        files = glob.glob(out_dir + '*BASE-BADM*')
        if len(results) > 0:
            assert files is not None
            assert 'BASE-BADM' in files[0]

        if files is not None:
            if len(files) > 0:
                os.remove(files[0])

        # Test if BIF files are downloaded
        results = act.discovery.ameriflux.download_ameriflux_data(
            user_id,
            user_email,
            data_product='BIF',
            data_policy=data_policy,
            data_variant=data_variant,
            site_ids=site_ids,
            agree_policy=True,
            intended_use=intended_use,
            description=description,
            out_dir=out_dir,
            **kwargs,
        )

        files = glob.glob(out_dir + '*BIF*xlsx')
        if len(results) > 0:
            assert files is not None
            assert 'BIF' in files[0]

        if files is not None:
            if len(files) > 0:
                for file in files:
                    os.remove(file)

        # Test for raised error if user_id is not a string
        pytest.raises(
            ValueError,
            act.discovery.ameriflux.download_ameriflux_data,
            3,
            user_email,
            data_product=data_product,
            data_policy=data_policy,
            data_variant=data_variant,
            site_ids=site_ids,
            agree_policy=True,
            intended_use=intended_use,
            description=description,
            out_dir=out_dir,
            **kwargs,
        )

        # Test for raised error if email is not a valid email format
        pytest.raises(
            ValueError,
            act.discovery.ameriflux.download_ameriflux_data,
            user_id,
            'foo',
            data_product=data_product,
            data_policy=data_policy,
            data_variant=data_variant,
            site_ids=site_ids,
            agree_policy=True,
            intended_use=intended_use,
            description=description,
            out_dir=out_dir,
            **kwargs,
        )

        # Test for raised error if agree_policy is False
        pytest.raises(
            ValueError,
            act.discovery.ameriflux.download_ameriflux_data,
            user_id,
            user_email,
            data_product=data_product,
            data_policy=data_policy,
            data_variant=data_variant,
            site_ids=site_ids,
            agree_policy=False,
            intended_use=intended_use,
            description=description,
            out_dir=out_dir,
            **kwargs,
        )

        # Test for raised error if data_policy is incorrect
        pytest.raises(
            ValueError,
            act.discovery.ameriflux.download_ameriflux_data,
            user_id,
            user_email,
            data_product=data_product,
            data_policy='foo',
            data_variant=data_variant,
            site_ids=site_ids,
            agree_policy=True,
            intended_use=intended_use,
            description=description,
            out_dir=out_dir,
            **kwargs,
        )

        # Test for raised error if site_id is not valid
        pytest.raises(
            ValueError,
            act.discovery.ameriflux.download_ameriflux_data,
            user_id,
            user_email,
            data_product=data_product,
            data_policy=data_policy,
            data_variant=data_variant,
            site_ids=['foo'],
            agree_policy=True,
            intended_use=intended_use,
            description=description,
            out_dir=out_dir,
            **kwargs,
        )

        # Test for raised error if intended_use category is not valid
        pytest.raises(
            ValueError,
            act.discovery.ameriflux.download_ameriflux_data,
            user_id,
            user_email,
            data_product=data_product,
            data_policy=data_policy,
            data_variant=data_variant,
            site_ids=site_ids,
            agree_policy=True,
            intended_use='foo',
            description=description,
            out_dir=out_dir,
            **kwargs,
        )

        # Test for warning if one of the site_ids doesn't exist
        with pytest.warns(UserWarning):
            act.discovery.ameriflux.download_ameriflux_data(
                user_id,
                user_email,
                data_product=data_product,
                data_policy=data_policy,
                data_variant=data_variant,
                site_ids=['US-CU1', 'foo'],
                agree_policy=True,
                intended_use=intended_use,
                description=description,
                out_dir=out_dir,
                **kwargs,
            )
