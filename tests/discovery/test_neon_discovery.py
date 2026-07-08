import os

import act


def test_neon():
    token = os.getenv('NEON_API')
    if token is not None:
        if len(token) == 0:
            return

        site_code = 'BARR'
        result = act.discovery.get_neon_site_products(token, site_code, print_to_screen=True)
        assert 'DP1.00002.001' in result
        assert result['DP1.00003.001'] == 'Triple aspirated air temperature'

        product_code = 'DP1.00002.001'
        result = act.discovery.get_neon_product_avail(
            token, site_code, product_code, print_to_screen=True
        )
        assert '2017-09' in result
        assert '2022-11' in result

        output_dir = os.path.join(os.getcwd(), site_code + '_' + product_code)
        result = act.discovery.download_neon_data(
            token, site_code, product_code, '2022-10', output_dir=output_dir
        )
        assert len(result) == 20
        assert any('readme' in r for r in result)
        assert any('sensor_position' in r for r in result)

        result = act.discovery.download_neon_data(
            token, site_code, product_code, '2022-09', end_date='2022-10', output_dir=output_dir
        )
        assert len(result) == 40
        assert any('readme' in r for r in result)
        assert any('sensor_position' in r for r in result)
