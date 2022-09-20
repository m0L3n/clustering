SELECT * FROM sales;         --6544
SELECT * FROM claims;        --2266
SELECT * FROM car_details;   --6544

--query 1:
-- The voucher budget for each selling month and what it represents compared to total revenues)
    CREATE OR REPLACE VIEW sales_view AS
    SELECT "merchant_id "AS merchant_id, country, car_id, TO_DATE(selling_date,'MM/DD/YYYY')AS selling_date,
           EXTRACT(MONTH FROM TO_DATE(selling_date,'MM/DD/YYYY' ))AS selling_month, EXTRACT(YEAR FROM TO_DATE(selling_date,'MM/DD/YYYY' ))AS selling_year, sell_price
    FROM sales;

    CREATE OR REPLACE VIEW claims_view AS
    SELECT claim_id, car_id, full_refunds, partial_refunds, voucher_amount,
        CASE    WHEN claim_status ='closed_fully_processed' AND estimation_transport_cost IS NOT NULL
                THEN SUM(voucher_amount + estimation_transport_cost)
                WHEN claim_status ='closed_fully_processed' AND estimation_transport_cost IS NULL
                THEN voucher_amount
                ELSE 0   END  AS voucher_budged, claim_status, free_transport, estimation_transport_cost
    FROM claims GROUP BY claim_id, car_id, full_refunds, partial_refunds, voucher_amount, claim_status, free_transport, estimation_transport_cost;

SELECT selling_month, selling_year, SUM(voucher_budged)AS voucher_budget, SUM(sell_price)AS revenues_per_month , (SUM(voucher_budged)/SUM(sell_price))AS vouchers_per_revenues
FROM claims_view JOIN sales_view ON claims_view.car_id = sales_view.car_id
GROUP BY selling_month, selling_year ORDER BY selling_year,selling_month;

--query 2:
-- The top 10 merchants that generated the most claims ever
SELECT  merchant_id, country, COUNT(claims_view.car_id) AS claims_count
FROM sales_view JOIN claims_view ON   claims_view.car_id = sales_view.car_id
GROUP BY merchant_id, country ORDER BY COUNT(claims_view.car_id) DESC LIMIT 10;

--query 3:
-- The radio system that generated the most claims for each selling month
        CREATE OR REPLACE VIEW claims_radio_system AS
        SELECT  radio_system, selling_month, selling_year, COUNT(radio_system)AS amount_rs_claims
        FROM claims_view    JOIN sales_view ON claims_view.car_id = sales_view.car_id
                    JOIN car_details ON claims_view.car_id = car_details."car_id "
        GROUP BY    selling_year, selling_month,  radio_system
        ORDER BY selling_year, selling_month, amount_rs_claims DESC;

WITH    t1 AS(SELECT MAX(amount_rs_claims)AS max_amount_rs_claims , selling_year, selling_month FROM claims_radio_system GROUP BY selling_year, selling_month),
        t2 AS (SELECT * FROM claims_radio_system )
    SELECT t2.radio_system, t1.max_amount_rs_claims,t2.selling_month , t2.selling_year
    FROM t1 JOIN t2 ON t1.selling_month = t2.selling_month and t1.selling_year = t2.selling_year and t1.max_amount_rs_claims = t2.amount_rs_claims;