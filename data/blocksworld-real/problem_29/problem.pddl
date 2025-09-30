(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        purple - block
        red - block
        green - block
        orange - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (clear yellow)
        (ontable purple)
        (clear purple)
        (ontable red)
        (on green red)
        (on orange green)
        (clear orange)
        (ontable blue)
        (clear blue)
        ; yellow
        ; purple
        ; red -> green -> orange
        ; blue
    )
    (:goal (and (on red yellow)
        (on purple red)
        (on green purple)
        (on orange green)
        (on blue orange)))
)
    