(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        yellow - block
        purple - block
        red - block
        green - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on yellow blue)
        (on purple yellow)
        (on red purple)
        (on green red)
        (on orange green)
        (clear orange)
        ; blue -> yellow -> purple -> red -> green -> orange
    )
    (:goal (and (on yellow red)
        (on purple yellow)
        (on green purple)
        (on orange green)
        (on blue orange)))
)
    